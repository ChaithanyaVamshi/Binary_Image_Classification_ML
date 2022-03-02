# Cats and Dogs Image Classification Using Machine Learning with Python
The Ultimate Guide for building Binary Image Classifier by applying Image Analysis and Pre-processing techniques on unseen photos of Dogs and Cats.

![image](https://user-images.githubusercontent.com/31254745/156273676-a601feea-a259-43c7-bf09-b862764fcca6.png)

## Introduction
There’s a strong belief that when it comes to working with unstructured data, especially image data, deep learning models are the best. Deep learning algorithms undoubtedly perform extremely well, but is that the only way to work with images?

Have you ever wondered what if there was a way, we could classify the images of cats and dogs using Machine Learning? That would be great right?

If we provide the right data and features, Machine learning models can perform adequately well and can even be used as an automated solution.

In this project, I will demonstrate and show how we can harness the power of Machine Learning and perform image classification using four popular machine learning algorithms like Random Forest Classifier, K-Nearest Neighbour (KNN) Classifier, Decision Tree Classifier, and Support Vector Machine (SVM).

## Problem Statement
The main objective of this task is to apply Machine learning algorithms to build Binary Image Classifier by applying Image Analysis and Pre-processing techniques on unseen photos of Dogs and Cats.

## Implementation of Binary Image Classification

1. Importing Libraries
2. Unzipping the Train and Test dataset
3. Creating Separate Train and Test dataframe
4. Checking Distribution of Label (Target Variable) in Train Data
5. Checking Shape of Train and Test Data
6. Feature Encoding - Custom Mapping "Label" Target Attribute
7. Creating Separate Train and Test folders to Load Images
8. Image Manipulation
9. Feature Extraction
10. Model Building
11. Model Evaluation
12. Predictions on Test Data

## Image Pre-processing on Train Data

### 1. Image Manipulation

- Gray Scaling: Grayscale is a range of monochromatic shades from black to white. Gray scaling in converting images to grayscale. 

- Resizing: Image resizing refers to either enlarging or shrinking images. Machine learning algorithms require the same size images during the learning and prediction phases. To convert all images to a common size, we need to define a base image size and resize images. 

- Smoothing/Blurring: Smoothing is commonly used for noise reduction in images. It reduces irrelevant details such as pixel regions that are small for the filter kernel size. In this task, I have applied the Gaussian Filtering technique which assigns the Gaussian weighted average of all the pixels under the kernel area as the central element value.

### 2. Feature Extraction

- Image Vectorisation: Method to convert an image to a vector is matrix vectorisation. Colour image vectorisation results in very long vectors which leads to a curse of dimensionality. Hence, the simplest solution is image gray scaling when compared to vectorisation.

- Edge Detection: Edge detection is an image processing technique that finds the boundaries of objects within images. In this task, I have applied Canny edge detection is an optimal algorithm for edge detection which helps in Noise reduction, finding intensity gradient of the image, filters out non-real edges and removing pixels that may not constitute edges

- HOG Feature Descriptor:  Histogram of Oriented Gradients feature descriptor counts the occurrences of gradient orientation in localised portions of an image as to features. It mainly focuses on the shape or the structure of objects. 

- Principle Component Analysis (PCA): PCA is a dimensionality reduction technique that uses Singular Value Decomposition of the data to project it to a lower-dimensional space.

## Model Building

1.	Support Vector Machine (SVM) Classifier	(M1 - M6)
2.	Decision Tree Classifier				(M7 - M12)
3.	K-Nearest Neighbour (KNN) Classifier		(M13 - M18)
4.	Random Forest Classifier 				(M19 - M24)

### Summary of Machine Learning Model Accuracy on Train and Validation Data

From all the Machine Learning Models, Random Forest Classifier (M22) with Gray scaling and HOG Feature has achieved the best Accuracy of 70% followed by Support Vector Machine (SVM) Classifier with Gray scaling and HOG Feature with 67% Accuracy.

![image](https://user-images.githubusercontent.com/31254745/156274561-87bc8510-dac6-4e38-8d43-8558c3faeb05.png)

![image](https://user-images.githubusercontent.com/31254745/156274623-03a55d19-e68e-47ed-b9b1-3b0c820c1240.png)

## Predictions on Test Dataset

Using the best ML Model Random Forest Classifier (M22), we will make predictions on test data and save predictions on .csv file name “test-predictions.csv”

![image](https://user-images.githubusercontent.com/31254745/156274969-f87a57cc-fbdb-47bb-9448-7b736c2af86e.png)


## Conclusion 

In this project, we discussed how to approach the binary Image classification problem by implementing four ML algorithms including Random Forest, KNN, Decision Tree, and Support Vector Machines. 

Out of these 4 ML Algorithms, Random Forest Classifier with HOG Feature extraction shows the best performance with 70% accuracy.

We can explore this work further by trying to improve the Machine Learning Image classification using hyperparameter tuning and Deep learning algorithms.



