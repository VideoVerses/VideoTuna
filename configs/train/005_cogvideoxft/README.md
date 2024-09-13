# cogvideo 

this branch use pytorch lightning framework to reorganize cogvideo training and inference. 

## environment
please follow the requirenments at `requrienemnt_cogVideo.txt`
## inference
please refer to the scripts for configuration.
```shell
bash configs/train/005_cogvideoxft/inference.sh
```

or 

```shell
bash shscripts/inference_cogvideo_pl.sh
```
## train
The results has been tested to be correct on a tiny dataset. 
please refer to the scripts for configuration. 

```shell
bash configs/train/005_cogvideoxft/run.sh
```

## implementation details.

train and inference lanucher, all configs are saved to `configs/train/005_cogvideoxft`.


### dataset 

the dataset follow original cogvideo format which has follow foramts:


```
.
├── labels
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── ...
```



> Tips: To build a fake dataset, simply duplicate the videos label files.


