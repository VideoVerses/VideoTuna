import torch
from torch.utils.data import Dataset


def from_file(path, low=None, high=None, **kwargs):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    prompts = lines[low:high]
    return prompts


class RLHF_PROMPT(Dataset):
    def __init__(
        self,
        data_root,
        resolution,
        video_length,
    ):
        self.data_root = data_root
        self.prompts = from_file(self.data_root)
        self.resolution = resolution
        self.video_length = video_length
        self.video_placeholder = torch.ones(
            3, self.video_length, *self.resolution
        )  # [c,t,h,w]
        print("RLHF PROMPT DATASET has ", len(self.prompts), " prompts")
        print(self.prompts)
        # import pdb;pdb.set_trace()

    def __getitem__(self, idx):
        ## to hack the input api, we return a meaninglingless frame to provideo noise shape
        prompt = self.prompts[idx]
        data = {
            "video": self.video_placeholder,
            "caption": prompt,
        }
        return data

    def __len__(self):
        return len(self.prompts)
