
This folder contains data processing scripts.

# Code structure:
```
VideoTuna/
└── tools/
    └── data_process/
        ├── scenecut.py
        ├── caption.py
        ├── xxx.py
        ├── scenecut.sh
        ├── caption.sh
        ├── xxx.sh
        └── README.md
```

# Instructions
**Scnencut**
```
python scenecut.py --vid_dir <videos path> --out_dir <output path> --num_process <number of process>
```
**Caption**
```
python caption.py \
--model_path <model path> \
--vid_dir <videos path> \
--out_dir <output path> \
--num_frame <number of sample frames> \
--num_process <number of processes> \
--mp_no <process NO.>
```
