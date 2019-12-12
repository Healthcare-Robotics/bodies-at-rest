# Bodies at Rest
## 3D Human Pose and Shape Estimation from a Pressure Image using Synthetic Data

<p align="center">
  <img width="98%" src="https://github.com/henryclever/bodies-at-rest/blob/master/docs/figures/intro_overview.JPG?raw=true" alt="None"/>
</p>

Paper:

Video: https://www.youtube.com/watch?v=UHBqw0BYWEw

Link to PressurePose dataset:

## What code is in here?

This repository: 

* Allows you to visualize both synthetic and real data in the PressurePose dataset. This is to help you get started. The synthetic dataset includes 206,000 fully labeled pressure images, meaning that each pressure image has a corresponding SMPL human mesh parameterized by body shape (10 PCA parameters), pose (69 joint angles), posture (6 DOF global transform), gender, height, and weight. The real dataset includes 1051 pressure images with co-registered point cloud data, RGB data, gender, height, and weight. 
* Has the code for PressureNet. That gives you a jumping point if you are interested in considering other architectures. There are switches inside of the PressureNet code to modify how it is trained, e.g. to include height and weight during training or not. 

## PressurePose dataset visualization
For the synthetic data, when you run the following code, you will see something like the pictures below. There are flags in the code that allow you to segment based on the limbs and also to cut out mesh vertices that aren't facing the camera. The camera is positioned in the synthetic dataset at the same location as the real one, so cutting out the non-camera facing vertices will allow you to better compare the synthetic data to the real point cloud data.



## PressureNet training
There are 3 steps to train PressureNet as implemented in the paper.
* Step 1: Train network 1 for 100 epochs using loss function 1. Run the following: 
* Step 2: Compute a new dataset that has spatial map reconstructions from the PMR output of network 1. Run the following:
* Step 3: Train network 2 for 100 epochs using loss function 2. Run the following:
The data can take a long time to load. Use the `--qt` flag to run a quick test on a small portion of the dataset to check for bugs. You can use an euler angle parameterization instead of the direction cosines in the SMPL model. Use the `--losstype 'anglesEU'` for that. 

## What other packages do I need?
* SMPL: A Skinned Multi-Person Linear Model - https://smpl.is.tue.mpg.de/
* PyRender - https://github.com/mmatl/pyrender
* Trimesh - https://github.com/mikedh/trimesh
* PyTorch - https://pytorch.org/
* PyTorch HMR - https://github.com/MandyMo/pytorch_HMR
* Matplotlib, PyGlet, some others .... 


## What code isn't here? 
The code for generating more synthetic data isn't here. I've got it spread across multiple repositories, so it would be challenging to make it decipherable. I'd also worry that users would have trouble getting it up and running because of its complexity; what I have now needs work to make it less challenging to install. 
