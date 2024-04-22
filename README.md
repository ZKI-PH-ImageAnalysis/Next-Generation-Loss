# Next Generation Loss
This repository contains the code for the new loss function found by using the Genetic Programming (GP) approach. 

The full paper can be downloded here: [[PDF]]([https://pages.github.com/](https://arxiv.org/abs/2404.12948))

![grafik](https://github.com/ZKI-PH-ImageAnalysis/Next-Generation-Loss/assets/107623498/40df7cc8-01cf-495a-96d4-30fd04d4bcf5)

# Example
The repository contains an example of how NGL can be used to train the InceptionV3 model on CIFAR-100 dataset. 

# Using NGL in your own project
The repository contains two implementations of the NGL loss: in pytorch and in tensorflow. Both versions can be used to train deep learning models for image classification. 

# Results
GP was used to conduct 5 different experiments in order to find loss functions applicable for classification tasks. The evaluation of the five found functions was performed by training ResNet50 and InceptionV3 on seven datasets, which differed by the number of images, classes, by the type of images (grayscale and RGB), and their sizes. ResNet50 was added to the initial experiments, as all the found functions were evaluated on InceptionV3 model during the GP search to rule out model specific properties. The top layers of both InceptionV3 and ResNet50 networks were not included, in both cases instead 9 new layers such as Flatten, BatchNormalization (×3), Dense (×3) and Dropout (×2) were added.

Both InceptionV3 and ResNet50 models were trained five times for each dataset. Obtained results were averaged for both models and compared to the averaged CE (with and without L2 regularization), focal and SCE losses for the same models. In the end, the percentage of accuracy improvement or decrease compared to CE was determined. A final value from the range [−0.5, 0.5] was considered as same performing, a value > +0.5% better performing and < −0.5% performing worse compared to CE.
Results obtained for all evaluated loss functions used to train models from scratch are given in the table below.


| Loss | Malaria    | Pcam    | Colorectal Histology    | CIFAR-10    | Fashion-MNIST   | CIFRA-100  | Caltech 101  | Mean  |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CE  | 94.0  | 69.4  | 88.9  | 92.8  | 94.0  | 68.2  | 72.5  | ±0  |
| SCE | -0.06 | +0.62 | -0.45 | -3.66 | -2.42 | -0.80 | +2.71 | -0.58  |
| Focal  | +0.34  | +1.03  | +2.89  | -0.64  | -0.02  | -2.14  | -0.78  | +0.10  |
| CE + L2  | +0.11  | -0.64  | +0.88  | +0.32  | +0.09  | +0.38  | +3.67  | +0.69  |
| f1  | -0.27  | +0.81  | +3.27  | -0.34  | +0.31  | -0.84  | -3.67  | -0.10  |
| f2  | -0.09  | +2.89  | -7.56  | -0.81  | +0.04  | +0.60  | -3.61  | -1.22  |
| f3  | -27.98  | +3.24  | -62.45  | -1.79  | -0.48  | -59.36  | -41.67  | -27.21  |
| f4  | -0.21  | -0.13  | -1.98  | -0.14  | 0.00  | -0.32  | -6.49  | -1.32  |
| f5 (NGL)  | -0.27  | +7.07  | +1.77  | +0.12  | +0.07  | +1.01  | +5.00  | +2.11  |


More information regarding the parameter settings and results can be found in the paper.
