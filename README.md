# CIFAR-10 Image Classification 🚀

This project demonstrates training a convolutional neural network (CNN) on the CIFAR-10 dataset using PyTorch.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Overview
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes. This project trains a CNN to classify these images.

## Dataset
Download the dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Model Architecture
The model includes:
- Convolutional layers
- Max-pooling layers
- Fully connected layers
- Dropout for regularization

Here is a visualization of the model architecture:
![Model Architecture](https://github.com/JillSunday/CIFAR-10-Classification-Model/blob/main/results/Model_Architecture.jpeg?raw=true)

## Results
- **Training Accuracy**: 95.05% (at 69 epochs)
- **Validation Accuracy**: 71.43%
- **Test Accuracy**: 71.43%

### Data Loading
The data was successfully loaded and preprocessed for training:
![Data Loading Successful](https://github.com/JillSunday/CIFAR-10-Classification-Model/blob/main/results/Dataloader_successful.jpeg?raw=true)

### Training Progress
The training and validation accuracy over epochs:
![Training Graph](results/Train_Graph.png)

## Usage
You can run this project either in **Google Colab** or on your **local machine**.

### Option 1: Run in Google Colab
1. Open the notebook in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JillSunday/CIFAR-10-Classification-Model/blob/main/notebooks/CIFAR_10_GROUP_PROJECT.ipynb)
2. Follow the steps in the notebook to train and evaluate the model.

### Option 2: Run on Your Local Machine
1. **Clone the repository**:
   ```bash
   git clone https://github.com/JillSunday/CIFAR-10-Classification-Model.git
   cd CIFAR-10-Classification-Model
2. **Install dependencies**:
    ```bash
   pip install -r requirements.txt
3. **Download the dataset**:
   Run the download_data.py script to download and preprocess the CIFAR-10 dataset:
   ```bash
   python scripts/download_data.py
4. **Train the model**:
   Run the train.py script to train the model:
   ```bash
   python scripts/train.py
5. **Test the model**:
   After training, run the test.py script to evaluate the model on the test set:
   ```bash
   python scripts/test.py

## Dependencies
PyTorch
Torchvision
NumPy
Matplotlib

## License
This project is licensed under the MIT License. See LICENSE for details.
   
    
