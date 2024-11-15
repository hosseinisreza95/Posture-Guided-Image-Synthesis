
## Posture-Guided Image Synthesis - Project

# Team Members (DISS):
1. Seyed Reza HOSSEINI (N 2401137)
2. Tanaz PIRIAEI (N 2320403)
3. Moslem PIRZAD (N 2317068)

# Project Overview
This project focuses on generating realistic images that follow specific human poses using a neural network and GAN (Generative Adversarial Network).
Our model learns to create images based on posture data extracted from skeleton poses, with the goal of achieving visually coherent images that follow given skeleton structures.
The methodology is inspired by the Everybody Dance Now paper by Chan et al., presented at ICCV 2019.

# Files and Structure
`GenNearest.py:` Core script for generating images based on the nearest skeletons.
`GenVanillaNN.py:` Core script for implementing the basic neural network.
`GenGan.py:` Core script for implementing the GAN model.
`DanceDemo.py:` Script for seeing the results of the models by choosing the number of the model.


# Model Details
Our model architecture includes adaptive convolution layers that transform skeleton pose data into images.
Training is guided by calculating the Mean Squared Error (MSE) between the generated and target images to improve accuracy.
The GAN model, after training for approximately 2000 epochs, learned to generate increasingly realistic images that closely align with the input poses.


# Running DanceDemo.py with Different Models
To use the `DanceDemo.py` script for generating images based on dance poses, you can specify the model type to control how the poses are processed. The `GEN_TYPE` variable in `DanceDemo.py` should be set to an integer (1 to 4) to select the desired model. Each model corresponds to a specific generation technique, as described below:

# Set GEN_TYPE = 1 for GenNeirest:
Uses the `nearest` skeleton-based generator. This approach selects and displays the nearest skeleton frames based on the target posture data.

# Set GEN_TYPE = 2 for GenVanillaNN with Skeleton Input:
This model uses a simple neural network `(VanillaNN)` trained to generate images directly from skeleton data.

# Set GEN_TYPE = 3 for GenVanillaNN with Image Input:
Another variant of the `VanillaNN` generator. This version uses additional image data alongside skeletons to enhance the generated output.

# Set GEN_TYPE = 4 for GenGAN:
Uses a `GAN` (Generative Adversarial Network) model for generation. This model leverages adversarial training to produce more realistic images.
Running DanceDemo.py


# After setting the `GEN_TYPE` to the desired model number, you can run the script as follows:

bash: python DanceDemo.py

During execution:
* The script reads frames from the source video (e.g., data/taichi1.mp4).
* Press q to exit the demo.
* Press n to skip 500 frames in the source video.
* The generated images will be displayed as combined frames with the source and target images side by side for comparison.