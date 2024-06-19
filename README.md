
# Depth Anything for Nuke

## Introduction

Introducing **Depth Anything** for **The Foundry's Nuke**. This project brings state-of-the-art **depth map** generation to your favorite compositing software.

**Depth Anything** is a neural network that creates accurate depth maps from single images, handling a wide range of subjects.

This tool is **natively integrated** within Nuke, providing a seamless and streamlined experience. It requires no external dependencies or complicated setup, enabling artists to leverage the cutting-edge AI model in their familiar workflow.

<div align="center">

[![author](https://img.shields.io/badge/by:_Rafael_Silva-red?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rafael-silva-ba166513/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>

https://github.com/rafaelperez/Depth-Anything-for-Nuke/assets/1684365/bad0e0d6-5468-408b-9f8c-ee9f186f7323

## Features

- **Simple interface**.
- **Efficient memory usage** - the high-quality model fits on most 8GB graphics cards.

## Compatibility

**Nuke 13.2v9+**, tested on **Linux** and **Windows**.

## Installation

1. Download and unzip the latest release from [here](https://github.com/rafaelperez/Depth-Anything-for-Nuke/releases).
2. Copy the extracted `Cattery` folder to `.nuke` or your plugins path.
3. In the toolbar, choose **Cattery > Update** or simply **restart** Nuke.

**Depth Anything** will then be accessible under the toolbar at **Cattery > Segmentation > Depth Anything**.

## Quick Start
Simply connect an input image to the **Depth Anything** node. Further controls and options will be added soon.

## Pre-trained models

The current release only includes the **Large** model, which is the only one that's suitable for visual effects work. However, the other models can be converted to `.cat` format by following the **compilation** steps below.

| Model | Params | Model Size | Inference Time on Nuke 13 | Nuke 14 |
|:-|-:|:-:|:-:|:-:|
| Depth-Anything-Small | 24.8M | - | - | - |
| Depth-Anything-Base | 97.5M | - | - | - |
| **Depth-Anything-Large** | 335.3M | 1.2Gb | - | - |

> *Tests under Rocky Linux 8, AMD Ryzen Threadripper 3960X 24-Core, NVidia GeForce RTX 3090.*

## Compiling the Model

To retrain or modify the model for use with **Nuke's CatFileCreator**, you'll need to convert it into the PyTorch format `.pt`. Below are the primary methods to achieve this:

### Cloud-Based Compilation (Recommended for Nuke 14+)

**Google Colaboratory** offers a free, cloud-based development environment ideal for experimentation or quick modifications. It's important to note that Colaboratory uses **Python 3.10**, which is incompatible with the **PyTorch version (1.6)** required by **Nuke 13**.

For those targetting **Nuke 14** or **15**, [Google Colaboratory](https://colab.research.google.com) is a convenient choice.

### Local Compilation (Required for Nuke 13+)

Compiling the model locally gives you full control over the versions of **Python**, **PyTorch**, and **CUDA** you use. Setting up older versions, however, can be challenging.

For **Nuke 13**, which requires **PyTorch 1.6**, using **Docker** is highly recommended. This recommendation stems from the lack of official PyTorch package support for **CUDA 11**.

Fortunately, Nvidia offers Docker images tailored for various GPUs. The Docker image version **20.07** is specifically suited for **PyTorch 1.6.0 + CUDA 11** requirements.

Access to these images requires registration on [Nvidia's NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

Once Docker is installed on your system, execute the following command to initiate a terminal within the required environment. You can then clone the repository and run `python sam_nuke.py` to compile the model.

```sh
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:20.07-py3
git clone https://github.com/rafaelperez/Depth-Anything-for-Nuke.git
cd Depth-Anything-for-Nuke
python nuke_dan.py
```
For projects targeting **Nuke 14+**, which requires **PyTorch 1.12**, you can use the following Docker image, version **22.05**:

`docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:22.05-py3`

For more information on selecting the appropriate Python, PyTorch, and CUDA combination, refer to [Nvidia's Framework Containers Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2020).

## License and Acknowledgments

**DepthAnything.cat** is licensed under the MIT License, and is derived from https://github.com/LiheYoung/Depth-Anything.

While the MIT License permits commercial use of **Depth Anything**, the dataset used for its training may be under a non-commercial license.

This license **does not cover** the underlying pre-trained model, associated training data, and dependencies, which may be subject to further usage restrictions.

Consult https://github.com/LiheYoung/Depth-Anything for more information on associated licensing terms.

**Users are solely responsible for ensuring that the underlying model, training data, and dependencies align with their intended usage of RIFE.cat.**

## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      booktitle={CVPR},
      year={2024}
}
```
