# Speech Emotion Recognition

This project focuses on recognizing emotions from speech using machine learning techniques. The dataset used for this project is the Toronto Emotional Speech Set (TESS). The project is implemented in a Jupyter notebook using Google Colab.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Speech Emotion Recognition (SER) is a challenging yet fascinating area of machine learning and artificial intelligence. This project aims to build a model that can classify the emotions expressed in speech. The emotions included in this dataset are angry, disgust, fear, happy, neutral, sad, and surprised.

## Dataset

The dataset used in this project is the **Toronto Emotional Speech Set (TESS)**, which contains audio recordings of 7 different emotions.

## Installation

To run this project, you need to have the following libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `librosa`
- `tensorflow`
- `kaggle`

You can install the required libraries using:

```sh
pip install -r requirements.txt

#Usage
1)Clone the repository:
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition

2)Upload your Kaggle API key:
Upload your kaggle.json file in the project directory. This file is necessary to download the dataset from Kaggle.

#Run the Jupyter Notebook:
Open the Emotion_Recognition.ipynb notebook in Jupyter or Google Colab and follow the steps to preprocess the data, train the model, and evaluate the results.

#Data Preprocessing
The notebook includes the following steps for data preprocessing:
Load the dataset.
Extract features from audio files using the librosa library.
Encode labels for the emotions.

#Model Training
The notebook includes the following steps for model training:
Split the data into training and testing sets.
Create and train an LSTM model using TensorFlow/Keras.

#Evaluation
After training the model, evaluate its performance using the test set. The notebook includes code for generating evaluation metrics and visualizations, such as accuracy, precision, recall, F1-score, and confusion matrix.

#Results
The results of the model training and evaluation are included in the notebook. The performance metrics such as accuracy, precision, recall, and F1-score are presented, along with confusion matrices and other relevant visualizations.

#Contributing
Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

#License
This project is licensed under the MIT License.
