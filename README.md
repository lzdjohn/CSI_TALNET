# ActionFormer: Localizing Moments of Actions with Transformers
说明：
./libs存放了模型代码，./libs/modeling/first_diff.py存放了一阶差分网络
## Introduction
This code repo implements Actionformer, one of the first Transformer-based model for temporal action localization --- detecting the onsets and offsets of action instances and recognizing their action categories. Without bells and whistles, ActionFormer achieves 71.0% mAP at tIoU=0.5 on THUMOS14, outperforming the best prior model by 14.1 absolute percentage points and crossing the 60% mAP for the first time. Further, ActionFormer demonstrates strong results on ActivityNet 1.3 (36.56% average mAP) and the more challenging EPIC-Kitchens 100 (+13.5% average mAP over prior works). Our paper is accepted to ECCV 2022 and an arXiv version can be found at [this link](https://arxiv.org/abs/2202.07925).

In addition, ActionFormer is the backbone for many winning solutions in the Ego4D Moment Queries Challenge 2022. Our submission in particular is ranked 2nd with a record 21.76% average mAP and 42.54% Recall@1x, tIoU=0.5, nearly three times higher than the official baseline. An arXiv version of our tech report can be found at [this link](https://arxiv.org/abs/2211.09074). We invite our audience to try out the code.
850030的数据，8500算作时间，30算作channel。通过提取特征 TC特征金字塔—> 得到特征；分出两个简单的branch网络，分别是class_head与regression_head。
视频中的fps参数，在信号中可以对应采样率；但信号数据怎么定义frame呢

Specifically, we adopt a minimalist design and develop a Transformer based model for temporal action localization, inspired by the recent success of Transformers in NLP and vision. Our method, illustrated in the figure, adapts local self-attention to model temporal context in untrimmed videos, classifies every moment in an input video, and regresses their corresponding action boundaries. The result is a deep model that is trained using standard classification and regression loss, and can localize moments of actions in a single shot, without using action proposals or pre-defined anchor windows.


## Code Overview
The structure of this code repo is heavily inspired by Detectron2. Some of the main components are
* ./libs/core: Parameter configuration module.
* ./libs/datasets: Data loader and IO module.
* ./libs/modeling: Our main model with all its building blocks.
* ./libs/utils: Utility functions for training, inference, and postprocessing.

## Installation
* Follow INSTALL.md for installing necessary dependencies and compiling the code.

## To Reproduce Our Results on THUMOS14
**Download Features and Annotations**
* Download *thumos.tar.gz* (`md5sum 375f76ffbf7447af1035e694971ec9b2`) from [this Box link](https://uwmadison.box.com/s/glpuxadymf3gd01m1cj6g5c3bn39qbgr) or [this Google Drive link](https://drive.google.com/file/d/1zt2eoldshf99vJMDuu8jqxda55dCyhZP/view?usp=sharing) or [this BaiduYun link](https://pan.baidu.com/s/1TgS91LVV-vzFTgIHl1AEGA?pwd=74eh).
* The file includes I3D features, action annotations in json format (similar to ActivityNet annotation format), and external classification scores.

**Details**: The features are extracted from two-stream I3D models pretrained on Kinetics using clips of `16 frames` at the video frame rate (`~30 fps`) and a stride of `4 frames`. This gives one feature vector per `4/30 ~= 0.1333` seconds.

**Unpack Features and Annotations**
* Unpack the file under *./data* (or elsewhere and link to *./data*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───data/
│    └───thumos/
│    │	 └───annotations
│    │	 └───i3d_features   
│    └───...
|
└───libs
│
│   ...
```

**Training and Evaluation**
* Train our ActionFormer with I3D features. This will create an experiment folder under *./ckpt* that stores training config, logs, and checkpoints.
```shell
python ./train.py ./configs/thumos_i3d.yaml --output reproduce
```
* [Optional] Monitor the training using TensorBoard
```shell
tensorboard --logdir=./ckpt/thumos_i3d_reproduce/logs
```
* Evaluate the trained model. The expected average mAP should be around 62.6(%) as in Table 1 of our main paper. **With recent commits, the expected average mAP should be higher than 66.0(%)**.
```shell
python ./eval.py ./configs/thumos_i3d.yaml ./ckpt/thumos_i3d_reproduce
```
* Training our model on THUMOS requires ~4.5GB GPU memory, yet the inference might require over 10GB GPU memory. We recommend using a GPU with at least 12 GB of memory.

**[Optional] Evaluating Our Pre-trained Model**

We also provide a pre-trained model for THUMOS 14. The model with all training logs can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1isG3bc1dG5-llBRFCivJwz_7c_b0XDcY/view?usp=sharing). To evaluate the pre-trained model, please follow the steps listed below.

* Create a folder *./pretrained* and unpack the file under *./pretrained* (or elsewhere and link to *./pretrained*).
* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───pretrained/
│    └───thumos_i3d_reproduce/
│    │	 └───thumos_reproduce_log.txt
│    │	 └───thumos_reproduce_results.txt
│    │   └───...    
│    └───...
|
└───libs
│
│   ...
```
* The training config is recorded in *./pretrained/thumos_i3d_reproduce/config.txt*.
* The training log is located at *./pretrained/thumos_i3d_reproduce/thumos_reproduce_log.txt* and also *./pretrained/thumos_i3d_reproduce/logs*.
* The pre-trained model is *./pretrained/thumos_i3d_reproduce/epoch_034.pth.tar*.
* Evaluate the pre-trained model.
```shell
python ./eval.py ./configs/thumos_i3d.yaml ./pretrained/thumos_i3d_reproduce/
```
* The results (mAP at tIoUs) should be

| Method            |  0.3  |  0.4  |  0.5  |  0.6  |  0.7  |  Avg  |
|-------------------|-------|-------|-------|-------|-------|-------|
| ActionFormer      | 82.13 | 77.80 | 70.95 | 59.40 | 43.87 | 66.83 |

```shell
python ./eval.py ./configs/ego4d_omnivore_egovlp.yaml ./pretrained/ego4d_omnivore_egovlp_reproduce/
```
* The results (mAP at tIoUs) should be

| Method                |  0.1  |  0.2  |  0.3  |  0.4  |  0.5  |  Avg  |
|-----------------------|-------|-------|-------|-------|-------|-------|
| ActionFormer (S)      | 20.09 | 17.45 | 14.44 | 12.46 | 10.00 | 14.89 |
| ActionFormer (O)      | 23.87 | 20.78 | 18.39 | 15.33 | 12.65 | 18.20 |
| ActionFormer (E)      | 26.84 | 23.86 | 20.57 | 17.19 | 14.54 | 20.60 |
| ActionFormer (S+E)    | 27.98 | 24.46 | 21.21 | 18.56 | 15.60 | 21.56 |
| ActionFormer (O+E)    | 27.99 | 24.94 | 21.94 | 19.05 | 15.98 | 21.98 |
| ActionFormer (S+O+E)  | 28.26 | 24.69 | 21.88 | 19.35 | 16.28 | 22.09 |

* The results (Recall@1x at tIoUs) should be

| Method                |  0.1  |  0.2  |  0.3  |  0.4  |  0.5  |  Avg  |
|-----------------------|-------|-------|-------|-------|-------|-------|
| ActionFormer (S)      | 52.25 | 45.84 | 40.60 | 36.58 | 31.33 | 41.32 |
| ActionFormer (O)      | 54.63 | 48.72 | 43.03 | 37.76 | 33.57 | 43.54 |
| ActionFormer (E)      | 59.53 | 54.39 | 48.97 | 42.75 | 37.12 | 48.55 |
| ActionFormer (S+E)    | 59.96 | 53.75 | 48.76 | 44.00 | 38.96 | 49.09 |
| ActionFormer (O+E)    | 61.03 | 54.15 | 49.79 | 45.17 | 39.88 | 49.99 |
| ActionFormer (S+O+E)  | 60.85 | 54.16 | 49.60 | 45.12 | 39.87 | 49.92 |
