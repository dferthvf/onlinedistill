# Online Self-Disllation 
This repo contains the code and [Slides](https://docs.google.com/presentation/d/1eCF1Ijya7bVu2fXUnvucS-WPuzYVSB84MQofq686LHM/edit?usp=sharing) for our project "Online Distillation for Few-Shot Learning". 

## Team Members
Sihan Liu (sihan@bu.edu)
Yixuan Zhang (yixuanz@bu.edu)
Yang Yu (yuyang00@bu.edu)

## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. However, it should be compatible with recent PyTorch versions >=0.4.0

## Download Data
The data we used here is preprocessed by the repo of [MetaOptNet](https://github.com/kjunelee/MetaOptNet), but we have
renamed the file. Our version of data can be downloaded from here:

[[DropBox]](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0)

## Pre-trained Models

[[DropBox]](https://www.dropbox.com/sh/6xt97e7yxheac2e/AADFVQDbzWap6qIGIHBXsA8ca?dl=0)

## Running

Exemplar commands for running the code can be found in `scripts/run.sh`.

### Training 

`python train_od.py --trial od_999_200 --model_path /path/to/save/your/model --tb_path /path/to/save/your/tensorboard_log --data_root /path/to/your/data --cosine`

### Testing 

`python eval_fewshot.py --model_path checkpoint/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_cosine_trial_od_999/ckpt_epoch_200.pth --data_root /path/to/your/model`


## References
**"Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results"** [Paper](https://arxiv.org/abs/1703.01780) [Github](https://github.com/CuriousAI/mean-teacher)
```
@article{TarvainenV17,
  author    = {Antti Tarvainen and
               Harri Valpola},
  title     = {Weight-averaged consistency targets improve semi-supervised deep learning
               results},
  journal   = {arXiv preprint abs:1703.01780},
  year      = {2017},
}
```

**"Rethinking few-shot image classification: a good embedding is all you need?"** [Paper](https://arxiv.org/abs/2003.11539),  [Project Page](https://people.csail.mit.edu/yuewang/projects/rfs/) 
```
@article{tian2020rethink,
  title={Rethinking few-shot image classification: a good embedding is all you need?},
  author={Tian, Yonglong and Wang, Yue and Krishnan, Dilip and Tenenbaum, Joshua B and Isola, Phillip},
  journal={arXiv preprint arXiv:2003.11539},
  year={2020}
}
```


## Acknowlegements
Part of the code is from [RFS](https://github.com/WangYueFt/rfs) repo.


