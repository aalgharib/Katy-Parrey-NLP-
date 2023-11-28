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
#Import file containing data
path = os.path.dirname(os.path.abspath(__file__))
filename = "Youtube02-KatyPerry (1).csv"
fullpath = os.path.join(path,filename)
katy_perry = pd.read_csv(fullpath, sep=',')
# =============================================================================
# Data Exploration
# =============================================================================
print(katy_perry.head(3))
print(katy_perry.shape)
print(katy_perry.info())
#Drop irrelevant columns
katy_perry=katy_perry[['CONTENT','CLASS']]
print(katy_perry.head(3))
# =============================================================================
# Data Visualization
# =============================================================================
plt.hist(katy_perry['CLASS'],bins=[-0.5,0.5,1.5], edgecolor=('black'))
plt.title("Distribution of Spam vs. Non-Spam Comments")
plt.xlabel('0=Non-Spam, 1=Spam')
plt.ylabel('Count')
plt.xticks([0,1])
plt.show()
# =============================================================================
# Pre-processing
# =============================================================================

import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer



count_vectorizer = CountVectorizer(stop_words='english')
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

