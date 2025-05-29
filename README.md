# CRC Image Classifier

## Project Overview

This project implements and compares several machine learning algorithms, namely k-Nearest Neighbours (KNN), Multilayer Perceptron (MLP), and Convolutional Neural Network (CNN) on a medical image classification task using the **PathMNIST** dataset. The study explores the entire machine learning pipeline including data exploration, preprocessing, model design, hyperparameter tuning, evaluation, and interpretation of results.

## Dataset

- **Source**: Subset of PathMNIST dataset from MedMNIST v2 collection ([link](https://www.nature.com/articles/s41597-022-01721-8)).
- **Description**: 28x28 RGB histopathological images of human tissues labeled into 9 classes (adipose, background, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma, colorectal adenocarcinoma epithelium).
- **Splits**:
  - Training: 32,000 images
  - Test: 8,000 images

## Project Structure

- **Jupyter Notebook (`.ipynb`)**: Contains code for data loading, preprocessing, model implementation, hyperparameter tuning, training, and evaluation.
- **PDF Report**: Provides a comprehensive description of the study, data exploration, methods, hyperparameter search, results, discussion, conclusion, and reflection.
- **Notebook PDF**: Rendered version of the notebook for easy viewing.

## Key Methods

- **k-Nearest Neighbours (KNN)**: Implemented as a non-parametric baseline using distance metrics (Manhattan and Euclidean) and weighting schemes (uniform and distance).
- **Multilayer Perceptron (MLP)**: Feedforward neural network with two hidden layers, ReLU activations, dropout for regularization, and hyperparameters tuned over hidden units, learning rate, optimizer, and batch size.
- **Convolutional Neural Network (CNN)**: Two convolutional layers followed by dense layers, leveraging spatial structure of images, with hyperparameters tuned over filters, dropout rate, learning rate, optimizer, and batch size.

## Results Summary

- **KNN**: Demonstrated basic classification performance with minimal preprocessing but higher runtime for distance calculations, especially with Manhattan distance.
- **MLP**: Showed improved performance over KNN with optimized architectures, but sensitive to learning rate and limited by shallow depth and training epochs.
- **CNN**: Achieved the highest classification accuracy, leveraging spatial hierarchies, but required longer training times.

## Key Learnings

- **Model Comparison**: Highlighted trade-offs between interpretability, accuracy, runtime, and model complexity.
- **Data Insights**: Importance of preprocessing, including normalization and stratified sampling, to handle class imbalance and improve model performance.
- **Practical Considerations**: Balancing runtime and performance, especially with limited computational resources.

## How to Run

1. Clone the repository.
2. Install the required Python libraries (`keras`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`).
3. Open the notebook (`.ipynb`) in Jupyter Notebook or Jupyter Lab.
4. Run cells sequentially to replicate the experiments and review the outputs.
5. Ensure the output of hyperparameter search is preserved for reproducibility.