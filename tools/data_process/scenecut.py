import argparse
import json
import multiprocessing as mp
import os
import subprocess
from typing import Any, List, Tuple, Union

import tqdm
from joblib import Parallel, delayed

# Standard PySceneDetect imports:
from scenedetect import SceneManager, VideoManager, open_video

# For content-aware scene detection:
from scenedetect.detectors import AdaptiveDetector, ContentDetector

# Standard PySceneDetect imports:
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.video_splitter import split_video_ffmpeg
from tqdm import tqdm


def get_video_resolution(video_path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        video_path,
    ]
    result = subprocess.check_output(command)
    res_data = json.loads(result)
    width = res_data["streams"][0]["width"]
    height = res_data["streams"][0]["height"]
    return [int(height), int(width)]


class MetadataDict:
    def __init__(self):
        self.metadata = {
            "basic": {
                "video_id": "",  # Type: str - clip belongs to which video
                "video_path": "",  # Type: str - source video path
                "video_duration": 0.0,  # Type: float (seconds)
                "video_resolution": [],  # Type: list [height, width]
                "video_fps": 0.0,  # Type: float
                "clip_id": "",  # Type: str
                "clip_path": "",  # Type: str
                "clip_duration": 0.0,  # Type: float (seconds)
                "clip_start_end_idx": [
                    0,
                    0,
                ],  # Type: list [int, int] - start and end frame indices in the source video (count from 0)
                "optimal_score": 0.0,  # Type: float
            },
            "scene": {
                "captions": "",  # Type: str - describes the content of the video clip
                "place": "",  # Type: str - keyword descriptions for the place
                "background": "",  # Type: str - keyword descriptions for the background
                "style": "",  # Type: str - keyword descriptions for the style
                "num_of_objects": 0,  # Type: int
                "objects": [  # Type: list - length is equal to num_of_objects
                    {
                        "category": "",  # Type: str - noun: human, dog, etc.
                        "action": "",  # Type: str - verb: run, dance, play guitar, etc.
                        "action_speed": "",  # Type: str - very slow/slow/medium/fast/very fast
                    }
                ],
            },
            "camera": {
                "view_scale": "",  # Type: str - long shot/full shot/medium shot/close-up shot/extreme close-up shot
                "movement": "",  # Type: str - static shot, pans and tilts shot, zoom in/zoom out/zoom in and zoom out
                "speed": "",  # Type: str - very slow/slow/medium/fast/very fast
            },
            "misc": {},  # Type: dict - additional miscellaneous metadata
        }

    def set_basic_info(
        self,
        index: int,
        video_id: str,
        video_path: str,
        scenes: List[List[FrameTimecode]],
        out_dir: str,
        optimal_score: float = None,
    ):
        scene = scenes[index]
        self.metadata["basic"]["video_id"] = video_id
        self.metadata["basic"]["video_path"] = os.path.join(
            os.path.basename(os.path.dirname(video_path)), os.path.basename(video_path)
        )
        self.metadata["basic"]["video_duration"] = scenes[-1][1].get_seconds()
        self.metadata["basic"]["video_resolution"] = get_video_resolution(video_path)
        self.metadata["basic"]["video_fps"] = scenes[0][0].get_framerate()
        self.metadata["basic"]["clip_id"] = f'{video_id}_{"%07d" % index}'
        self.metadata["basic"]["clip_path"] = f"{self.metadata['basic']['clip_id']}.mp4"
        self.metadata["basic"]["clip_duration"] = (scene[1] - scene[0]).get_seconds()
        self.metadata["basic"]["clip_start_end_idx"] = [
            scene[0].get_frames(),
            scene[1].get_frames(),
        ]
        self.metadata["basic"]["optimal_score"] = optimal_score

    def _set_basic_info(
        self,
        video_id: str,
        video_path: str,
        video_duration: float,
        video_resolution: List[int],
        video_fps: int,
        clip_id: str,
        clip_path: str,
        clip_duration: float,
        clip_start_end_idx: List[int],
        optimal_score: float,
    ):
        self.metadata["basic"]["video_id"] = video_id
        self.metadata["basic"]["video_path"] = video_path
        self.metadata["basic"]["video_duration"] = video_duration
        self.metadata["basic"]["video_resolution"] = video_resolution
        self.metadata["basic"]["video_fps"] = video_fps
        self.metadata["basic"]["clip_id"] = clip_id
        self.metadata["basic"]["clip_path"] = clip_path
        self.metadata["basic"]["clip_duration"] = clip_duration
        self.metadata["basic"]["clip_start_end_idx"] = clip_start_end_idx
        self.metadata["basic"]["optimal_score"] = optimal_score

    def set_scene_info(
        self,
        captions: str,
        place: str,
        background: str,
        style: str,
        num_of_objects: int,
        objects: List[dict],
    ):
        self.metadata["scene"]["captions"] = captions
        self.metadata["scene"]["place"] = place
        self.metadata["scene"]["background"] = background
        self.metadata["scene"]["style"] = style
        self.metadata["scene"]["num_of_objects"] = num_of_objects
        self.metadata["scene"]["objects"] = objects

    def set_camera_info(self, view_scale: str, movement: str, speed: str):
        self.metadata["camera"]["view_scale"] = view_scale
        self.metadata["camera"]["movement"] = movement
        self.metadata["camera"]["speed"] = speed

    def set_misc_info(self, misc_info: dict):
        self.metadata["misc"] = misc_info

    def load_from_dict(self, metadata_dict: dict):
        if "basic" in metadata_dict:
            self._set_basic_info(**metadata_dict["basic"])
        if "scene" in metadata_dict:
            try:
                self.set_scene_info(**metadata_dict["scene"])
            except:
                self.set_scene_info("", "", "", "", 0, [])
        if "camera" in metadata_dict:
            try:
                self.set_camera_info(**metadata_dict["camera"])
            except:
                self.set_camera_info("", "", "")
        if "misc" in metadata_dict:
            try:
                self.set_misc_info(metadata_dict["misc"])
            except:
                self.set_misc_info({})

    def get_metadata(self):
        return self.metadata

    def get_value(self, section: str, key: str) -> Any:
        if section in self.metadata and key in self.metadata[section]:
            return self.metadata[section][key]
        else:
            return None

    def update_value(
        self, section: str, key: str, value: Union[str, int, float, List[Any], dict]
    ) -> bool:
        if section in self.metadata and key in self.metadata[section]:
            self.metadata[section][key] = value
            return True
        else:
            return False

    def to_dict(self) -> dict:
        return self.metadata


def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list()


def find_breakpoint(metadata_list):
    vid_visited = dict()
    vid_finished = []
    for clip in metadata_list:
        meta_clip = MetadataDict()
        meta_clip.load_from_dict(clip)
        vid_id = os.path.basename(meta_clip.get_value("basic", "video_path"))
        if vid_id not in vid_visited:
            vid_visited[vid_id] = [meta_clip.get_value("basic", "video_duration"), 0.0]
        vid_visited[vid_id][1] += meta_clip.get_value("basic", "clip_duration")
        if vid_visited[vid_id][0] - vid_visited[vid_id][1] < 0.5:
            vid_finished.append(vid_id)
    return vid_finished


def main(vid_dir, out_dir, file_list):
    threshold = 30.0

    with open(os.path.join(out_dir, "metadata.json"), "a") as out_file:
        for vid_file in tqdm(file_list):
            try:
                if "." in vid_file:
                    vid_name, vid_ext = vid_file.rsplit(".", 1)
                    os.makedirs(f"{out_dir}/{vid_name}", exist_ok=True)
                else:
                    continue
                if vid_ext in [
                    "mp4",
                    "avi",
                    "mkv",
                    "mov",
                    "wmv",
                    "flv",
                    "webm",
                    "mpeg",
                    "mpg",
                ]:
                    vid_path = os.path.join(vid_dir, vid_file)
                    scenes = find_scenes(vid_path, threshold=threshold)
                    for index in range(len(scenes)):
                        metadata = MetadataDict()
                        metadata.set_basic_info(
                            index,
                            video_id=vid_name,
                            video_path=vid_path,
                            scenes=scenes,
                            out_dir=out_dir,
                        )
                        split_video_ffmpeg(
                            vid_path,
                            [scenes[index]],
                            f"{out_dir}/{vid_name}/{metadata.get_value('basic', 'clip_id')}.mp4",
                        )

                        output_metadata = json.dumps(metadata.to_dict())
                        out_file.write(output_metadata + ", ")
            except subprocess.CalledProcessError as e:
                print("FFmpeg error: ", e.stderr, " :  ", vid_path)
                continue
            except Exception as e:
                print("An error occurred:", str(e))
                continue


def run__process(vid_dir, out_dir, num_process):
    if not os.path.isabs(vid_dir):
        vid_dir = os.path.abspath(vid_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_json_path = os.path.join(out_dir, "metadata.json")
    finished_list = []
    if os.path.exists(out_json_path):
        if os.path.getsize(out_json_path) > 1:
            with open(out_json_path, "r") as oj:
                try:
                    line = oj.readlines()
                    processed_list = json.loads(
                        line[0][:-3] + line[0][-3:].replace(", ", " ]")
                    )
                    finished_list = find_breakpoint(processed_list)
                except:
                    finished_list = []

    file_list = os.listdir(vid_dir)
    file_list = list(set(file_list) - set(finished_list))
    if len(file_list) == 0:
        return
    return_flag = 1
    for file in file_list:
        if file.rsplit(".", 1)[-1] in [
            "mp4",
            "avi",
            "mkv",
            "mov",
            "wmv",
            "flv",
            "webm",
            "mpeg",
            "mpg",
        ]:
            return_flag = 0
            break
    if return_flag:
        return

    if len(finished_list) > 0:
        with open(out_json_path, "r") as oj:
            line = oj.readlines()
            out = line[0][:-2] + line[0][-2:].replace("]", ", ")
        with open(out_json_path, "w") as oj:
            oj.write(out)
    else:
        with open(out_json_path, "w") as oj:
            oj.write("[")

    chunk_size = (
        len(file_list) // num_process
        if len(file_list) >= num_process
        else len(file_list)
    )
    chunks = [
        file_list[i : i + chunk_size] for i in range(0, len(file_list), chunk_size)
    ]

    # Use joblib to allocate processes on CPUs
    Parallel(n_jobs=num_process)(
        delayed(main)(vid_dir, out_dir, chunk) for chunk in chunks
    )
    # from concurrent.futures import ProcessPoolExecutor
    # with ProcessPoolExecutor(max_workers=num_process) as executor:
    #     futures = []
    #     for chunk in chunks:
    #         future = executor.submit(main, vid_dir, out_dir, chunk)
    #         futures.append(future)
    #
    #     # Get results of all tasks
    #     for future in futures:
    #         result = future.result()

    with open(out_json_path, "r") as oj:
        line = oj.readlines()
        out = line[0][:-2] + " ]"
    with open(out_json_path, "w") as oj:
        json.dump(json.loads(out), oj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vid_dir",
        type=str,
        default="/project/vgenfmod/itsc-kftse-movies/movies/National-Geographic",
    )
    parser.add_argument("--out_dir", type=str, default="/project/llmsvgen/pengjun/tmp")
    parser.add_argument("--num_process", type=str)
    args = parser.parse_args()

    run__process(args.vid_dir, args.out_dir, int(args.num_process))
