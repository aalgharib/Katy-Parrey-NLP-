# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 00:06:46 2023

@author: becky
"""
import numpy as np
import os
import pandas as pd
import nltk
import matplotlib.pyplot as plt

# Import file containing data
path = os.path.dirname(os.path.abspath(__file__))
filename = "Youtube02-KatyPerry (1).csv"
fullpath = os.path.join(path, filename)
katy_perry = pd.read_csv(fullpath, sep=",")
# =============================================================================
# Data Exploration
# =============================================================================
print(katy_perry.head(3))
print(katy_perry.shape)
print(katy_perry.info())
# Drop irrelevant columns
katy_perry = katy_perry[["CONTENT", "CLASS"]]
print(katy_perry.head(3))
# =============================================================================
# Data Visualization
# =============================================================================
plt.hist(katy_perry["CLASS"], bins=[-0.5, 0.5, 1.5], edgecolor=("black"))
plt.title("Distribution of Spam vs. Non-Spam Comments")
plt.xlabel("0=Non-Spam, 1=Spam")
plt.ylabel("Count")
plt.xticks([0, 1])
plt.show()
# =============================================================================
# Pre-processing
# =============================================================================

import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


count_vectorizer = CountVectorizer(stop_words="english")
train_tc = count_vectorizer.fit_transform(katy_perry)

print("\nDimensions of  data:", train_tc.shape)

tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
type(train_tfidf)

# Transform input data using count vectorizer
input_tc = count_vectorizer.transform(katy_perry)
type(input_tc)
print(input_tc)
# Transform vectorized data using tfidf transformer
input_tfidf = tfidf.transform(input_tc)
type(input_tfidf)
print(input_tfidf)

# =============================================================================
# Model Building & Training
# =============================================================================
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB

# Shuffle the dataset
katy_perry_shuffled = katy_perry.sample(frac=1, random_state=99)

# Separate features and labels
X = katy_perry_shuffled["CONTENT"]
y = katy_perry_shuffled["CLASS"]

# Set the percentage for training data
train_percentage = 0.75
split_index = int(train_percentage * len(X))

# Split the dataset into training and testing sets using iloc
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Vectorize and transform training data
X_train_tc = count_vectorizer.transform(X_train)
X_train_tfidf = tfidf.transform(X_train_tc)

# Fit a Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tfidf, y_train)

# Cross-validate the model on the training data using 5-fold
kf = KFold(n_splits=5, shuffle=True, random_state=99)
cross_val_scores = cross_val_score(
    naive_bayes_classifier, X_train_tfidf, y_train, cv=kf
)

# Print the mean results of model accuracy
print("Cross Validated Accuracy: ", np.mean(cross_val_scores))

# =============================================================================
# Model Testing
# =============================================================================
