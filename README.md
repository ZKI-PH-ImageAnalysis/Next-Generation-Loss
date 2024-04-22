# Next Generation Loss (NGL)
This repository contains the code for the new loss function found by using the Genetic Programming (GP) approach. 

The full paper can be downloded [here](https://arxiv.org/abs/2404.12948).

![grafik](https://github.com/ZKI-PH-ImageAnalysis/Next-Generation-Loss/assets/107623498/40df7cc8-01cf-495a-96d4-30fd04d4bcf5)

# Example
The repository contains an example of how NGL can be used to train the InceptionV3 model on CIFAR-100 dataset. 

# Using NGL in your own project
The repository contains two implementations of the NGL loss: in pytorch and in tensorflow. Both versions can be used to train deep learning models for image classification. 

# Results
NGL was evaluated on seven datasets, which differed by the number of images, classes, by the type of images (grayscale and RGB), and their sizes. 


| Loss | Malaria    | Pcam    | Colorectal Histology    | CIFAR-10    | Fashion-MNIST   | CIFAR-100  | Caltech 101  | Mean  |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CE  | 94.0  | 69.4  | 88.9  | 92.8  | 94.0  | 68.2  | 72.5  | ±0  |
| SCE | -0.06 | +0.62 | -0.45 | -3.66 | -2.42 | -0.80 | +2.71 | -0.58  |
| Focal  | **+0.34**  | +1.03  | **+2.89**  | -0.64  | -0.02  | -2.14  | -0.78  | +0.10  |
| CE + L2  | +0.11  | -0.64  | +0.88  | **+0.32**  | **+0.09**  | +0.38  | +3.67  | +0.69  |
| f5 (NGL)  | -0.27  | **+7.07**  | +1.77  | +0.12  | +0.07  | **+1.01**  | **+5.00**  | **+2.11**  |


See paper for details.
