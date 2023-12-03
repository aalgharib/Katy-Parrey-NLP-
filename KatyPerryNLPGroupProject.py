import numpy as np
import os
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

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


# Pre-processing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer



# =============================================================================
# Pre-processing
# =============================================================================
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Vectorize and transform training data
count_vectorizer = CountVectorizer(stop_words="english")
X_train_tc = count_vectorizer.fit_transform(katy_perry["CONTENT"])
X_train_tfidf = TfidfTransformer().fit_transform(X_train_tc)

count_vectorizer = CountVectorizer(stop_words="english")

# Vectorize and transform training data
X_train_tc = count_vectorizer.fit_transform(katy_perry["CONTENT"])
X_train_tfidf = TfidfTransformer().fit_transform(X_train_tc)

# Model Building & Training
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB




# =============================================================================
# Model Building & Training
# =============================================================================
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB

# =============================================================================
# Model Building & Training
# =============================================================================

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
X_train_tfidf = TfidfTransformer().fit_transform(X_train_tc)

# Fit a Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tfidf, y_train)

# Cross-validate the model on the training data using 5-fold
kf = KFold(n_splits=5, shuffle=True, random_state=99)
cross_val_scores = cross_val_score(
    naive_bayes_classifier, X_train_tfidf, y_train, cv=kf
)

# Print the mean results of model accuracy

print("Cross Validated Accuracy with 5 k-folds: ", np.mean(cross_val_scores))


# Cross-validate the model on the training data using 9-fold
kf = KFold(n_splits=9, shuffle=True, random_state=99)
cross_val_scores = cross_val_score(
    naive_bayes_classifier, X_train_tfidf, y_train, cv=kf
)

# Print the mean results of model accuracy
print("Cross Validated Accuracy with 9 k-folds: ", np.mean(cross_val_scores))

print("Cross Validated Accuracy: ", np.mean(cross_val_scores))


# =============================================================================
# Model Testing
# =============================================================================

# Vectorize and transform test data
X_test_tc = count_vectorizer.transform(X_test)
X_test_tfidf = TfidfTransformer().fit_transform(X_test_tc)

# Predictions on the test set
y_pred = naive_bayes_classifier.predict(X_test_tfidf)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy: {:.2%}".format(accuracy))


# =============================================================================
# New comments
# =============================================================================

new_comments = [
    "I love Katy Perry's music! 😍",
    "The concert was amazing, such a great performance!",
    "Her voice is so beautiful, a true talent.",
    "Can't wait for the next album to drop!",
    "👉👉 Free iPhone Giveaway! Click the link now! 📱🆓",
    "Earn $1000 a day working from home! Check out this amazing opportunity!",
]

# Vectorize and transform the new comments using the same vectorizer and transformer
new_comments_tc = count_vectorizer.transform(new_comments)
new_comments_tfidf = TfidfTransformer().fit_transform(new_comments_tc)

# Predictions on the new comments
new_comments_predictions = naive_bayes_classifier.predict(new_comments_tfidf)

# Display results
for comment, prediction in zip(new_comments, new_comments_predictions):

    print(f"Comment: '{comment}' - Predicted Class: {prediction}")



