import argparse
import os
import zipfile
from pathlib import Path

from tqdm import tqdm


def unzip_vript(zip_file: Path, output_dir: Path):
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for file in zip_ref.namelist():
            zip_ref.extract(file, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="./Vript/vript_short_videos_clips_unzip"
    )
    parser.add_argument(
        "--zip_folder", type=str, default="./Vript/vript_short_videos_clips"
    )
    parser.add_argument("-s", "--start_idx", type=int, default=0)
    parser.add_argument("-e", "--end_idx", type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    zip_folder = Path(args.zip_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_list = sorted(os.listdir(str(zip_folder)))

    with tqdm(total=len(zip_list)) as pbar:
        for idx, zip_file in enumerate(zip_list):
            pbar.update(1)
            pbar.set_description(f"Processing {zip_file}")

            if idx < args.start_idx:
                continue
            if idx >= args.end_idx:
                break

            unzip_vript(zip_folder / zip_file, output_dir)
