# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .wan_i2v_A14B import i2v_A14B
from .wan_t2v_1_3B import t2v_1_3B
from .wan_t2v_A14B import t2v_A14B

# Legacy Wan2.1 task name aliases (VideoTuna configs / poetry scripts).
t2v_14B = t2v_A14B
i2v_14B = i2v_A14B
t2i_14B = copy.deepcopy(t2v_A14B)
t2i_14B.__name__ = "Config: Wan T2I 14B"

WAN_CONFIGS = {
    "t2v-A14B": t2v_A14B,
    "i2v-A14B": i2v_A14B,
    # Wan2.1 / VideoTuna legacy task names
    "t2v-14B": t2v_14B,
    "t2v-1.3B": t2v_1_3B,
    "i2v-14B": i2v_14B,
    "t2i-14B": t2i_14B,
}

SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
    "704*1280": (704, 1280),
    "1280*704": (1280, 704),
    "1024*704": (1024, 704),
    "704*1024": (704, 1024),
}

MAX_AREA_CONFIGS = {
    "720*1280": 720 * 1280,
    "1280*720": 1280 * 720,
    "480*832": 480 * 832,
    "832*480": 832 * 480,
    "704*1280": 704 * 1280,
    "1280*704": 1280 * 704,
    "1024*704": 1024 * 704,
    "704*1024": 704 * 1024,
}

SUPPORTED_SIZES = {
    "t2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "i2v-A14B": ("720*1280", "1280*720", "480*832", "832*480"),
    # Legacy Wan2.1 task names
    "t2v-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "t2v-1.3B": ("480*832", "832*480"),
    "i2v-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "t2i-14B": tuple(SIZE_CONFIGS.keys()),
}
