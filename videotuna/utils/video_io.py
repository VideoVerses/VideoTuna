"""Video frame sampling and decoding with PyAV / torchcodec fallbacks.

Install torchcodec via the optional ``video-fast`` extra::

    poetry install -E cuda --with training -E video-fast
"""

from __future__ import annotations

import importlib.util
import random
from typing import Literal, Optional, Sequence, Union

import av
import numpy as np
import torch
from einops import rearrange

VideoBackend = Literal["auto", "av", "torchcodec"]


def sample_frame_indices(
    total_frames: int,
    num_frames: int,
    frame_interval: int = 1,
    begin_index: Optional[int] = None,
) -> np.ndarray:
    """Sample frame indices matching TemporalRandomCrop randomness."""
    sample_length = num_frames * frame_interval
    rand_end = max(0, total_frames - sample_length - 1)
    if begin_index is None:
        begin_index = random.randint(0, rand_end)
    end_index = min(begin_index + sample_length, total_frames)
    if end_index - begin_index < num_frames:
        raise ValueError(
            f"The video has not enough frames. total={total_frames}, "
            f"need sample_length={sample_length}"
        )
    return np.linspace(begin_index, end_index - 1, num_frames, dtype=int)


def _open_video_container(
    video_path: str,
) -> tuple[av.container.InputContainer, av.VideoStream]:
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    return container, stream


def _stream_frame_count(stream: av.VideoStream) -> int:
    if stream.frames and stream.frames > 0:
        return int(stream.frames)
    if stream.duration and stream.average_rate:
        return int(stream.duration * stream.time_base * stream.average_rate)
    return 0


def _count_frames_by_decode(
    container: av.container.InputContainer, stream: av.VideoStream
) -> int:
    count = 0
    for _ in container.decode(stream):
        count += 1
    return count


def get_video_frame_count(video_path: str) -> int:
    """Return total frame count using PyAV metadata or demux fallback."""
    container, stream = _open_video_container(video_path)
    try:
        count = _stream_frame_count(stream)
        if count > 0:
            return count
        return _count_frames_by_decode(container, stream)
    finally:
        container.close()


def get_video_fps(video_path: str) -> float:
    """Return average frame rate for a video file."""
    container, stream = _open_video_container(video_path)
    try:
        if stream.average_rate:
            return float(stream.average_rate)
        if stream.base_rate:
            return float(stream.base_rate)
        return 0.0
    finally:
        container.close()


def get_frame_timestamp(video_path: str, index: int) -> tuple[float, float]:
    """Return (start, end) PTS in seconds for a frame index (decord-compatible)."""
    container, stream = _open_video_container(video_path)
    try:
        if index < 0:
            if stream.duration and stream.time_base:
                end = float(stream.duration * stream.time_base)
                return (0.0, end)
            fps = float(stream.average_rate) if stream.average_rate else 1.0
            count = _stream_frame_count(stream) or _count_frames_by_decode(
                container, stream
            )
            return (0.0, count / fps)

        fps = float(stream.average_rate) if stream.average_rate else 1.0
        start = index / fps
        end = (index + 1) / fps
        return (start, end)
    finally:
        container.close()


def _resize_frame_chw(
    frame: torch.Tensor, width: Optional[int], height: Optional[int]
) -> torch.Tensor:
    if width is None and height is None:
        return frame
    import cv2

    h, w = frame.shape[1], frame.shape[2]
    target_w = width if width is not None else w
    target_h = height if height is not None else h
    if target_w == w and target_h == h:
        return frame
    arr = frame.permute(1, 2, 0).numpy()
    resized = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(resized).permute(2, 0, 1)


def _read_av(
    video_path: str,
    indices: Sequence[int],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> torch.Tensor:
    idx_list = [int(i) for i in indices]
    if not idx_list:
        raise ValueError("indices must not be empty")

    wanted = set(idx_list)
    max_idx = max(idx_list)

    container, stream = _open_video_container(video_path)
    frames: dict[int, torch.Tensor] = {}
    try:
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx in wanted:
                arr = frame.to_ndarray(format="rgb24")
                tensor = torch.from_numpy(arr).permute(2, 0, 1)
                frames[frame_idx] = _resize_frame_chw(tensor, width, height)
            if frame_idx >= max_idx and len(frames) == len(wanted):
                break
    finally:
        container.close()

    missing = [i for i in idx_list if i not in frames]
    if missing:
        raise ValueError(
            f"Video {video_path} has fewer decodable frames than requested; "
            f"missing indices: {missing[:5]}"
        )
    return torch.stack([frames[i] for i in idx_list])


def _torchcodec_available() -> bool:
    return importlib.util.find_spec("torchcodec") is not None


def _resolve_auto_backends() -> list[str]:
    if _torchcodec_available():
        return ["torchcodec", "av"]
    return ["av"]


def _read_torchcodec(video_path: str, indices: Sequence[int]) -> torch.Tensor:
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(video_path, device="cpu")
    idx = [int(i) for i in indices]
    frames = decoder.get_frames_at(indices=idx)
    data = frames.data
    if data.ndim == 4 and data.shape[-1] in (1, 3, 4):
        return rearrange(data, "t h w c -> t c h w")
    return data


def read_video_frames(
    video_path: str,
    indices: Sequence[int],
    backend: VideoBackend = "auto",
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> torch.Tensor:
    """Decode selected frames as TCHW uint8 tensor."""
    backends: list[str]
    if backend == "auto":
        backends = _resolve_auto_backends()
    else:
        backends = [backend]

    last_error: Optional[Exception] = None
    for name in backends:
        try:
            if name == "av":
                return _read_av(video_path, indices, width=width, height=height)
            if name == "torchcodec":
                return _read_torchcodec(video_path, indices)
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Failed to decode {video_path} with backends {backends}"
    ) from last_error


class _NumpyBatch:
    """Mimics decord NDArray.asnumpy() for vendored Wan call sites."""

    def __init__(self, data: Union[np.ndarray, torch.Tensor]):
        if isinstance(data, torch.Tensor):
            self._data = data.cpu().numpy()
        else:
            self._data = data

    def asnumpy(self) -> np.ndarray:
        return self._data


class AvVideoReader:
    """Decord-compatible video reader backed by PyAV."""

    def __init__(
        self,
        video_path: str,
        ctx=None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_threads: int = 1,
    ):
        del ctx, num_threads  # CPU-only; threading handled by PyAV AUTO
        self.video_path = video_path
        self.width = width
        self.height = height
        self._frame_count: Optional[int] = None
        self._fps: Optional[float] = None

    def _ensure_metadata(self) -> None:
        if self._frame_count is None:
            self._frame_count = get_video_frame_count(self.video_path)
        if self._fps is None:
            self._fps = get_video_fps(self.video_path)

    def __len__(self) -> int:
        self._ensure_metadata()
        assert self._frame_count is not None
        return self._frame_count

    def get_avg_fps(self) -> float:
        self._ensure_metadata()
        assert self._fps is not None
        return self._fps

    def get_batch(self, indices: Sequence[int]) -> _NumpyBatch:
        frames = read_video_frames(
            self.video_path,
            indices,
            backend="av",
            width=self.width,
            height=self.height,
        )
        # TCHW uint8 -> THWC for decord compatibility
        thwc = rearrange(frames, "t c h w -> t h w c").numpy()
        return _NumpyBatch(thwc)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.get_batch([index]).asnumpy()[0]

    def get_frame_timestamp(self, index: int) -> tuple[float, float]:
        return get_frame_timestamp(self.video_path, index)

    def seek(self, index: int) -> None:
        """No-op kept for decord API compatibility."""
        del index


def init_video_worker() -> None:
    """Call once per DataLoader worker before decoding (no-op for PyAV)."""
