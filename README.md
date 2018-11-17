# Handwritten-Digit-Recognition

This is the project of classification of handwritten digits. Here I have taken an input image of handwritten numbers and then I have used a machine learning algorithm to identify each of these numbers actually are. 

At first, I have trained the classifier with the datasets of 5000 handwritten digits ranging between 0 and 9 (500 for each digit). Each digit is a 20x20 image. So my first step is to split this image into 5000 different digits. For each digit, I flatten it into a single row with 400 pixels. That is my feature set, i.e. intensity values of all pixels. It is the simplest feature set we can create. I have used first 350 samples of each digit as train_data, and next 150 samples as test_data.

In this project I have used k-Nearest Neighbour(kNN) algorithm as a machine learning algorithm. kNN is one of the simplest of classification algorithms available for supervised learning. The idea is to search for closest match of the test data in feature space.

This particular program gave me an accuracy of 93.47%.

## Requires:
   * OpenCv in Python
   * Numpy
   
## Input Image:
![numbers](https://user-images.githubusercontent.com/40036314/48663507-57567a00-eab7-11e8-822e-ac80eace819b.jpg)

## Output Images:

![screenshot from 2018-11-17 22-24-16](https://user-images.githubusercontent.com/40036314/48663591-6db10580-eab8-11e8-8e65-eec9b80fcefd.png)
![screenshot from 2018-11-17 22-24-42](https://user-images.githubusercontent.com/40036314/48663592-76094080-eab8-11e8-850e-cf5be45f9471.png)
![screenshot from 2018-11-17 22-24-53](https://user-images.githubusercontent.com/40036314/48663597-80c3d580-eab8-11e8-80af-ef6c10476395.png)
![screenshot from 2018-11-17 22-25-04](https://user-images.githubusercontent.com/40036314/48663599-89b4a700-eab8-11e8-8da1-279c46fd606d.png)
![screenshot from 2018-11-17 22-25-19](https://user-images.githubusercontent.com/40036314/48663603-920ce200-eab8-11e8-8050-398dadcd120e.png)

