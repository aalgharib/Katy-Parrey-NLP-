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
path = "C:/Users/becky/Documents/Software Engineering/Semester 3/COMP237-Introduction to AI/NLPProject/"
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



