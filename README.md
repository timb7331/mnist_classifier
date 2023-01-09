# mnist_classifier
a simple mnist classifier build using python and pytorch that allows the user to draw a number on a canvas and pass it through a Pytorch MNIST Classifier

![Alt-Text](https://i.imgur.com/vpkxD7J.png)



![Python](https://img.shields.io/badge/-Python-000?&logo=Python)
![PyTorch](https://img.shields.io/badge/-PyTorch-000?&logo=PyTorch)

# Requirements
- Python
- Pytorch
- Pillow
- Cuda GPU for faster Network Training.

# Info
I only tested the classifier with 14 epochs and my NVIDIA RTX 2080Ti using CUDA. It should work fine on CPU though. 
It might make some mistakes (eg. sometimes predict a 7 instead of a 1)

# Setup 
1. Download the code and install the dependencies using pip or conda.
2. Run the Script
3. A window should appear, click the retrain button to train the MNIST Classifier Model (Note: This will download the MNIST Dataset to your computer)
4. After the training is done, you should now be able to write a number on the canvas and use the classifier.
5. Have fun :) 
