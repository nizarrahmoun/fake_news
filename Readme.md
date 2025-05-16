# 📰 Fake News Detection

A machine learning project that builds a binary classification model to detect whether a news article is real or fake.

## Overview

This project uses Natural Language Processing (NLP) and machine learning techniques to classify news articles as either real or fake. The model achieves 99% accuracy using Logistic Regression with TF-IDF features.

## Dataset

The dataset consists of two CSV files located in the `News_dataset` folder:
- `Fake.csv` - Contains fake news articles
- `True.csv` - Contains real news articles

## Requirements

- Python 3.x
- Required packages:
  - pandas
  - numpy
  - nltk
  - scikit-learn

Install dependencies:
```bash
pip install nltk scikit-learn pandas numpy
```

## Project Structure

- `fake_news_detection_dataset_ready.ipynb` - Main Jupyter notebook containing:
  1. Data loading and preprocessing
  2. Text cleaning and TF-IDF vectorization
  3. Model training (Logistic Regression)
  4. Model evaluation

## Model Performance

The model achieves excellent performance metrics:
- Accuracy: 99%
- Precision: 98-99%
- Recall: 98-99%
- F1-score: 98-99%

## Usage

1. Clone the repository
2. Install the required packages
3. Run the Jupyter notebook `fake_news_detection_dataset_ready.ipynb`

## Data Preprocessing

The text preprocessing pipeline includes:
- Removing URLs
- Removing special characters and numbers
- Converting to lowercase
- Removing stopwords
- TF-IDF vectorization with 5000 features

## Model Training

- The dataset is split into 80% training and 20% validation sets
- A Logistic Regression classifier is trained on the TF-IDF features
- The model is evaluated using confusion matrix and classification report