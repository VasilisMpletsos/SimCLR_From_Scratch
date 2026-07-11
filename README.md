# SimCLR_From_Scratch

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#)

## Introduction

This repo is a deep dive to joint embedding world and more especially to the [SimCLR Paper](https://arxiv.org/abs/2002.05709) <i>(A Simple Framework for Contrastive Learning of Visual Representations)</i> written by T. Chen, S. Kornblith, M. Norouzi and G. Hinton.

Here i create from scratch in [Pytorch](https://docs.pytorch.org/docs/2.13/index.html) the networks, loss functions and the whole pipeline for **Contrastive Learning** training.

<center>
<img src="./assets/contrastive_learning.jpg" alt="Contrastive Learning" width="600">
</center>

I split the work in the distinct parts in order to have cleaner structure and follow more easily the paper step by step, so the stages are:

1. [Image Augementation](#image-augementation)
2. [Base Encoder](#base-encoder)
3. [Projection Head](#projection-head)
4. [Loss Functions](#loss-functions)
5. [Pipeline](#pipeline)

Finally i conduct some experiments in order to evaluate the work and validate the results.

## Dataset
As the base dataset i use the [evanarlian/imagenet_1k_resized_256](https://huggingface.co/datasets/evanarlian/imagenet_1k_resized_256) which is the **ImageNet already transforemd to 256 x 256 pixels** for smaller compute requirements, and because [datasets](https://github.com/huggingface/datasetss) support streaming mode which is very usefull for my case so i dont download the whole dataset locally.

## Image Augementation
As specified in the paper the augmentation that stood out and was selected in the end was the following 3 image augmentation steps:
    
1. [Random Crop](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomCrop.html)
2. [Resize](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.Resize.html)
3. [Color Distortion](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html)
4. [Gaussian Blur](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.GaussianBlur.html)
    

Below i am presenting a random sample from the dataset where the 1st image is the real one and the 4 remaining are examples where [image_augmentation](./src/augmentation.py) is performed.
<center>
<img src="./assets/Image Augmentaitons.jpg" alt="Image Augmentations" width="800">
</center>

***Important**: In the paper it is stated that <i>"Color histograms alone suffice to distinguish images. Neural net may exploint this shortcut to solve the predictive task"</i> and that is the reason that color distortion and gaussian blur are used, in order to mitigate this "hacky way" in and let the model learn generalizable features.*
## Base Encoder
## Projection Head
## Loss Functions
## Pipeline
