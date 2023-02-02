# Zero-shot instance segmentation

<!----------------------------------------------------------------------------------------------------------------------
#
#   Description
#
# --------------------------------------------------------------------------------------------------------------------->
This is a ROS packages for Zero-shot instance segmentation based on Mask-RCNN and CLIP

<!----------------------------------------------------------------------------------------------------------------------
#
#   Table of Contents
#
# --------------------------------------------------------------------------------------------------------------------->
## Table of Contents
  * [Requirement](#requirement)
  * [Components](#component)
  * [Example](#example)
  * [Installation](#installation)
  * [Usage](#usage)
  * [TODO](#todo)
  * [References](#references)


<!----------------------------------------------------------------------------------------------------------------------
#
#   Requirement
#
# --------------------------------------------------------------------------------------------------------------------->
## Requirement
* Required
  * Ubuntu 20.04
  * ROS Noetic
  * Python 3.8
  * Detectron2 0.6
  * CLIP 
  * CUDA 11.3

<!----------------------------------------------------------------------------------------------------------------------
#
#   Components
#
# --------------------------------------------------------------------------------------------------------------------->
## Components
Zero-shot instance segmentation code is consisting of "Image Agnostic Segmentation" module and "Zero-shot classifier".

In this code, Mask-RCNN trained by Anas Gouda et al. [1], and CLIP [2] is used as zero-shot classifier.

I implemented Mask-RCNN with Detectron2 [3]

<!----------------------------------------------------------------------------------------------------------------------
#
#   Example
#
# --------------------------------------------------------------------------------------------------------------------->
## Example
This movie is a example of prediction result.

![demo](demo/demo.gif)

<!----------------------------------------------------------------------------------------------------------------------
#
#   Installation
#
# --------------------------------------------------------------------------------------------------------------------->
## Installation
```shell
# install detectron2 (Your CUDA must be 11.3)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# download mask-rcnn checkpoint
mkdir -p ./image_agnostic_segmentation_ros/models
cd ./image_agnostic_segmentation_ros/models
wget https://tu-dortmund.sciebo.de/s/ISdLcDMduHeW1ay/download  -O FAT_trained_Ml2R_bin_fine_tuned.pth
# install CLIP
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

<!----------------------------------------------------------------------------------------------------------------------
#
#   Installation
#
# --------------------------------------------------------------------------------------------------------------------->
## Usage
### How to change prompts.
Configuration file for prompts is in ./image_agnostic_segmentation_ros/prompts-config/

If you want change prompts, you need to edit the file named "prompts.txt"

### How to build ros node.
```shell
roslaunch image_agnostic_segmentation_ros image_agnostic_segmentation.launch
```

<!----------------------------------------------------------------------------------------------------------------------
#
#   TODO
#
# --------------------------------------------------------------------------------------------------------------------->
## TODO
Check for bugs in CLIP and Mask-RCNN code (on going)

Check the prediction result when input commpressed image.

Define ros msg to publish prediction result.

<!----------------------------------------------------------------------------------------------------------------------
#
#   TODO
#
# --------------------------------------------------------------------------------------------------------------------->
## References
[1] https://github.com/AnasIbrahim/image_agnostic_segmentation

[2] https://github.com/openai/CLIP

[3] https://github.com/facebookresearch/detectron2