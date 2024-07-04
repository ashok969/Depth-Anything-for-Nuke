
# Depth Anything V2 for Nuke

## Introduction

Introducing **Depth Anything** for **The Foundry's Nuke**. This project brings state-of-the-art **depth map** generation to your favorite compositing software.

**Depth Anything (V2)** is a neural network that creates accurate depth maps from single images, handling a wide range of subjects.

This tool is **natively integrated** within Nuke, providing a seamless and streamlined experience. It requires no external dependencies or complicated setup, enabling artists to leverage the cutting-edge AI model in their familiar workflow.

<div align="center">

[![author](https://img.shields.io/badge/by:_Rafael_Silva-red?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rafael-silva-ba166513/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>

https://github.com/rafaelperez/Depth-Anything-for-Nuke/assets/1684365/bad0e0d6-5468-408b-9f8c-ee9f186f7323

## Features

- **Simple interface**: No 3D tracking setup required
- **High detail**: Captures relatively difficult edges
- **Efficient memory usage**: Downrez option allows good results on most 6GB graphics cards
- **CPU compatible**: Can be batched to CPU render farms

## My other Nuke gizmos
You may also be interested in my other Nuke Cattery nodes or gizmos. Feel free to check them out:

### ✨ AI-powered nodes
- **[RIFE for Nuke](https://github.com/rafaelperez/RIFE-for-Nuke)**: Advanced AI-driven retiming and optical flow
- **[SegmentAnything for Nuke](https://github.com/rafaelperez/Segment-Anything-for-Nuke)**: Cutting-edge AI object segmentation
- **[VITMatte for Nuke](https://github.com/rafaelperez/ViTMatte-for-Nuke):** AI-powered natural edge detail extraction

### Advanced effects
- **[Guided Blur/Refine Edge](https://www.nukepedia.com/gizmos/filter/guided-blur-refine-edge):** High-speed edge detail extraction using BlinkScript

## Compatibility

**Nuke 13.2v9+**, tested on **Linux** and **Windows**.

## Installation

1. Download and unzip the latest release from [here](https://github.com/rafaelperez/Depth-Anything-for-Nuke/releases).
2. Copy the extracted `Cattery` folder to `.nuke` or your plugins path.
3. In the toolbar, choose **Cattery > Update** or simply **restart** Nuke.

**Depth Anything V2** will then be accessible under the toolbar at **Cattery > Depth Estimation > DepthAnythingV2**.

### 🐾 Extra Steps for Nuke 13

4. Add the path for ***Depth Anything V2** to your `init.py`:
``` py
import nuke
nuke.pluginAddPath('./Cattery/DepthAnythingV2')
```
5. Add an menu item to the toolbar in your `menu.py`:

``` py
import nuke
toolbar = nuke.menu("Nodes")
toolbar.addCommand('Cattery/Depth Estimation/DepthAnythingV2', 'nuke.createNode("DepthAnythingV2")', icon="DepthAnythingV2.png")
```

## Quick Start
Connect an input image to the **Depth Anything** node. If necessary, adjust the **Near** and **Far** depth values to scale the Z-depth values accordingly.


## Options

<img src="https://github.com/rafaelperez/Depth-Anything-for-Nuke/assets/1684365/508733c2-f3ab-479d-bc24-b73331ed7900" width="640">

- **Use GPU if available:** Utilize GPU acceleration when possible.

- **Bypass sRGB Conversion:**  Skips sRGB color space conversion for input images.

- **View:** Selects the output display mode:
  - **Preview (False Color):** Displays the depth map in false colors for easy visualization.
  - **Final Output:** Output the Z-depth channel into the original input.

- **Downrez:** Reduces the resolution of the input image to optimize memory usage and processing speed.

- **Far:** Sets the maximum depth value in the scene, representing the farthest objects.

- **Near:** Defines the minimum depth value, corresponding to the closest objects in the scene.

- **Invert Map:** Reverses the depth values, making near objects appear far and vice versa.


## Pre-trained models

The current release includes the **V2_Small** model, which is the best commercially allowed model according to the original project's licensing terms. However, more capable models can be converted to `.cat` format by following the **compilation** steps below.

| Model | Params | Model Size | Inference Time on Nuke 13 | Nuke 14 |
|:-|-:|:-:|:-:|:-:|
| **Depth-Anything_V2-Small** | 24.8M | - | - | - |
| Depth-Anything_V2-Base | 97.5M | - | - | - |
| Depth-Anything_V2-Large | 335.3M | 1.2Gb | - | - |

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

Once **Docker** is installed on your system, execute the following command to initiate a terminal within the required environment. You can then clone the repository and run `python nuke_dan.py` to compile the model.

```sh
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:20.07-py3
git clone https://github.com/rafaelperez/Depth-Anything-for-Nuke.git
cd Depth-Anything-for-Nuke

# Compiles the default model, V2 small, half precision.
python nuke_dan.py

# Compiles the V2 large model, half precision.
python nuke_dan.py --version v2 --model-size vitl --half

# Check the available models and options with
python nuke_dan.py --help
```
For projects targeting **Nuke 14+**, which requires **PyTorch 1.12**, you can use the following Docker image, version **22.05**:

`docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:22.05-py3`

For more information on selecting the appropriate Python, PyTorch, and CUDA combination, refer to [Nvidia's Framework Containers Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2020).

## License and Acknowledgments

This repository, **DepthAnythingV2.cat** is licensed under the MIT License, and is derived from https://github.com/LiheYoung/Depth-Anything and https://github.com/DepthAnything/Depth-Anything-V2.

> [!IMPORTANT]
> DepthAnything original project pre-trained have different licenses. **As of July 7, 2024**:
> - Depth-Anything-V1-models are under the Apache-2.0 license (commercial use allowed). 
> - Depth-Anything-V2-Small model is under the Apache-2.0 license (commercial use allowed). 
> - Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license. (commercial use **not** allowed).  

While the MIT License permits commercial use of **Depth Anything**, the dataset used for its training may be under a non-commercial license.

This license **does not cover** the underlying pre-trained model, associated training data, and dependencies, which may be subject to further usage restrictions.

**Always refer to** https://github.com/LiheYoung/Depth-Anything and https://github.com/DepthAnything/Depth-Anything-V2 for the most up-to-date information on associated licensing terms.

> [!WARNING]
> **Users are solely responsible** for ensuring that the underlying model, training data, and dependencies align with their intended usage of DepthAnything.cat.

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```
