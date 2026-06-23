import sys
from pathlib import Path

sys.path.append(".")

import unittest

import videotuna.data.transforms as transforms
from videotuna.data.datasets import DatasetFromCSV

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_VIDEO_CSV = REPO_ROOT / "videotuna/data/anno_files/toy_video_dataset.csv"
TOY_IMAGE_CSV = REPO_ROOT / "videotuna/data/anno_files/toy_image_dataset.csv"
TOY_VIDEOS_DIR = REPO_ROOT / "videotuna/data/toy_videos"
TOY_IMAGES_DIR = REPO_ROOT / "videotuna/data/toy_images"


def _use_dummy_video(transform_video):
    if not TOY_VIDEOS_DIR.exists():
        transform_video.transforms[0] = transforms.LoadDummyVideo(
            (100, 100), probs_fail=0.5
        )


def _use_dummy_image(transform_image):
    if not TOY_IMAGES_DIR.exists():
        transform_image.transforms[0] = transforms.LoadDummyImage(probs_fail=0.5)


def _has_toy_images():
    return TOY_IMAGE_CSV.is_file()


class TestDatasets(unittest.TestCase):
    def test_video_dataset_from_csv(self):
        transform_video = transforms.get_transforms_video()
        _use_dummy_video(transform_video)
        dataset = DatasetFromCSV(
            TOY_VIDEO_CSV,
            str(TOY_VIDEOS_DIR),
            transform={"video": transform_video},
        )
        for i in range(min(5, len(dataset))):
            print(dataset[i].keys())
            self.assertFalse("height" in dataset[i].keys())
            self.assertFalse("width" in dataset[i].keys())
            self.assertGreater(dataset[i]["fps"], 0)

        print(f"len(dataset): {len(dataset)}")
        self.assertEqual(len(dataset), 128)
        self.assertEqual(dataset[0]["video"].shape[2], 256)

        transform_video.transforms[0] = transforms.LoadDummyVideo(probs_fail=0.4)
        dataset = DatasetFromCSV(
            TOY_VIDEO_CSV,
            str(TOY_VIDEOS_DIR),
            transform={"video": transform_video},
        )
        for i in range(min(5, len(dataset))):
            print(dataset[i].keys())
            self.assertFalse("height" in dataset[i].keys())
            self.assertFalse("width" in dataset[i].keys())
            self.assertGreater(dataset[i]["fps"], 0)

    def test_video_dataset_wo_transforms_from_csv(self):
        dataset = DatasetFromCSV(
            TOY_VIDEO_CSV,
            str(TOY_VIDEOS_DIR),
        )
        if not TOY_VIDEOS_DIR.exists():
            transform_video = dataset.transform["video"]
            transform_video.transforms[0] = transforms.LoadDummyVideo(probs_fail=0.5)
            dataset = DatasetFromCSV(
                TOY_VIDEO_CSV,
                transform={"video": transform_video},
            )
        for i in range(min(5, len(dataset))):
            print(dataset[i].keys())
            self.assertFalse("height" in dataset[i].keys())
            self.assertFalse("width" in dataset[i].keys())
            self.assertGreater(dataset[i]["fps"], 0)

        print(f"len(dataset): {len(dataset)}")
        self.assertEqual(len(dataset), 128)
        self.assertEqual(dataset[0]["video"].shape[2], 256)

    @unittest.skipUnless(_has_toy_images(), "toy image annotations not available")
    def test_image_dataset_from_csv(self):
        transform_image = transforms.get_transforms_image()
        _use_dummy_image(transform_image)
        dataset = DatasetFromCSV(
            TOY_IMAGE_CSV,
            str(TOY_IMAGES_DIR),
            transform={"image": transform_image},
        )
        for i in range(min(5, len(dataset))):
            print(dataset[i].keys())
            self.assertFalse("height" in dataset[i].keys())
            self.assertFalse("width" in dataset[i].keys())
            self.assertEqual(dataset[i]["fps"], 0)

        print(f"len(dataset): {len(dataset)}")
        self.assertEqual(len(dataset), 16)
        self.assertEqual(dataset[0]["video"].shape[2], 256)

    @unittest.skipUnless(_has_toy_images(), "toy image annotations not available")
    def test_multi_res(self):
        # Test Video
        transform_video = transforms.get_transforms_video()
        _use_dummy_video(transform_video)
        dataset = DatasetFromCSV(
            TOY_VIDEO_CSV,
            str(TOY_VIDEOS_DIR),
            transform={"video": transform_video},
            use_multi_res=True,
        )
        for i in range(min(5, len(dataset))):
            print(dataset[i].keys())
            self.assertTrue("height" in dataset[i].keys())
            self.assertTrue("width" in dataset[i].keys())
            self.assertTrue("fps" in dataset[i].keys())

        print(f"len(dataset): {len(dataset)}")
        self.assertEqual(len(dataset), 128)
        self.assertEqual(dataset[0]["video"].shape[2], 256)

        # Test Image
        transform_image = transforms.get_transforms_image()
        _use_dummy_image(transform_image)
        dataset = DatasetFromCSV(
            TOY_IMAGE_CSV,
            str(TOY_IMAGES_DIR),
            transform={"image": transform_image},
            use_multi_res=True,
        )
        for i in range(min(5, len(dataset))):
            print(dataset[i].keys())
            self.assertTrue("height" in dataset[i].keys())
            self.assertTrue("width" in dataset[i].keys())
            self.assertEqual(dataset[i]["fps"], 0)

        print(f"len(dataset): {len(dataset)}")
        self.assertEqual(len(dataset), 16)
        self.assertEqual(dataset[0]["video"].shape[2], 256)

    @unittest.skipUnless(_has_toy_images(), "toy image annotations not available")
    def test_concat_dataset_from_csv(self):
        transform_video = transforms.get_transforms_video()
        _use_dummy_video(transform_video)

        transform_image = transforms.get_transforms_image()
        _use_dummy_image(transform_image)
        dataset = DatasetFromCSV(
            [
                TOY_VIDEO_CSV,
                TOY_IMAGE_CSV,
            ],
            [str(TOY_VIDEOS_DIR), str(TOY_IMAGES_DIR)],
            transform={"video": transform_video, "image": transform_image},
        )
        for i in range(min(5, len(dataset))):
            print(dataset[i].keys())
            self.assertFalse("height" in dataset[i].keys())
            self.assertFalse("width" in dataset[i].keys())
            self.assertTrue("fps" in dataset[i].keys())

        print(f"len(dataset): {len(dataset)}")
        self.assertEqual(len(dataset), 144)
        self.assertEqual(dataset[0]["video"].shape[2], 256)

    def test_anno_wo_meta_info(self):
        transform_video = transforms.get_transforms_video()
        _use_dummy_video(transform_video)
        dataset = DatasetFromCSV(
            TOY_VIDEO_CSV,
            str(TOY_VIDEOS_DIR),
            transform={"video": transform_video},
            use_multi_res=True,
        )
        data_list = dataset.data_list
        for i, data_item in enumerate(data_list):
            data_list[i] = {"path": data_item["path"], "caption": data_item["caption"]}

        dataset.data_list = data_list
        for i in range(min(5, len(dataset))):
            print(dataset[i].keys())
            self.assertTrue("height" in dataset[i].keys())
            self.assertTrue("width" in dataset[i].keys())
            self.assertTrue("fps" in dataset[i].keys())

        print(f"len(dataset): {len(dataset)}")
        self.assertEqual(len(dataset), 128)
        self.assertEqual(dataset[0]["video"].shape[2], 256)

    def test_anno_wo_meta_info_wo_multi_res(self):
        transform_video = transforms.get_transforms_video()
        _use_dummy_video(transform_video)
        dataset = DatasetFromCSV(
            TOY_VIDEO_CSV,
            str(TOY_VIDEOS_DIR),
            transform={"video": transform_video},
            use_multi_res=False,
        )
        data_list = dataset.data_list
        for i, data_item in enumerate(data_list):
            data_list[i] = {"path": data_item["path"], "caption": data_item["caption"]}

        dataset.data_list = data_list
        for i in range(min(5, len(dataset))):
            print(dataset[i].keys())
            self.assertFalse("height" in dataset[i].keys())
            self.assertFalse("width" in dataset[i].keys())
            self.assertTrue("fps" in dataset[i].keys())

        print(f"len(dataset): {len(dataset)}")
        self.assertEqual(len(dataset), 128)
        self.assertEqual(dataset[0]["video"].shape[2], 256)

    def test_video_dataset_from_csv_with_split(self):
        transform_video = transforms.get_transforms_video()
        _use_dummy_video(transform_video)

        # Test Training Dataset
        train_dataset = DatasetFromCSV(
            TOY_VIDEO_CSV,
            str(TOY_VIDEOS_DIR),
            transform={"video": transform_video},
            split_val=True,
        )
        for i in range(min(5, len(train_dataset))):
            print(train_dataset[i].keys())
            self.assertFalse("height" in train_dataset[i].keys())
            self.assertFalse("width" in train_dataset[i].keys())
            self.assertGreater(train_dataset[i]["fps"], 0)

        print(f"len(dataset): {len(train_dataset)}")
        self.assertLessEqual(len(train_dataset), 128)
        self.assertEqual(train_dataset[0]["video"].shape[2], 256)

        # Test Validation Dataset
        val_dataset = DatasetFromCSV(
            TOY_VIDEO_CSV,
            str(TOY_VIDEOS_DIR),
            transform={"video": transform_video},
            train=False,
            split_val=True,
        )
        for i in range(min(5, len(val_dataset))):
            print(val_dataset[i].keys())
            self.assertFalse("height" in val_dataset[i].keys())
            self.assertFalse("width" in val_dataset[i].keys())
            self.assertGreater(val_dataset[i]["fps"], 0)

        print(f"len(dataset): {len(val_dataset)}")
        self.assertLessEqual(len(val_dataset), 128)
        self.assertEqual(val_dataset[0]["video"].shape[2], 256)
        # Check train + validation lengths sum to total sample count
        self.assertEqual(len(train_dataset) + len(val_dataset), 128)


if __name__ == "__main__":
    unittest.main()
