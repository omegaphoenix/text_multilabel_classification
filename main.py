#!/bin/python
# Example of multilabel classification using sklearn

# Resources
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# http://scikit-learn.org/stable/modules/multiclass.html
# https://stackoverflow.com/questions/10526579/use-scikit-learn-to-classify-into-multiple-categories

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

def validate_line(quote_and_label):
    return len(quote_and_label) == 2

# Read file into two arrays
# 1. input_quotes - list of all the quotes
# 2. input_labels - list of list of labels e.g. [[label0, label1], [label0, label2], [label1], ...]
def read_file(file_name):
    line_idx = 0
    # Array of quotes
    input_quotes = []
    # Array of array of labels (multiple labels per quote)
    input_labels = []
    with open(file_name, 'r', errors='replace') as file:
        for line in file:
            line_idx = line_idx + 1
            if line_idx == 1:
                continue
            quote_and_label = line.strip('\n"').split('\t')
            if (validate_line(quote_and_label)):
                quote = quote_and_label[0].strip('"')
                labels = quote_and_label[1].strip('"').split(',')
                input_quotes.append(quote)
                input_labels.append(labels)
    return input_quotes, input_labels

# Print each line and the predicted values
def predict_labels(classifier, X_train, y_train, X_test, y_test, target_names):
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    for item, predictions in zip(X_test, predicted):
        labels = [i for i, x in enumerate(predictions) if x]
        print('%s => %s' % (item, ', '.join(target_names[x] for x in labels)))

# Precision and recall for each label
def evaluate_model(classifier, X_train, y_train, X_test, y_test, target_names):
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    print(metrics.classification_report(y_test, predicted, target_names=target_names))


def main():
    data_path = './quotes.txt'
    input_quotes, input_labels = read_file(data_path)

    # Represent labels as binary arrays
    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(input_labels)
    target_names = list(mlb.classes_)

    print(target_names)
    X_train, X_test, y_train, y_test = train_test_split(input_quotes, binary_labels, test_size=0.33, random_state=16, stratify=binary_labels)

    # Pipeline for classification
    classifier = Pipeline([
        ('vectorizer', CountVectorizer()), # bag of words
        ('tfidf', TfidfTransformer()), # normalize
        ('clf', OneVsRestClassifier(LinearSVC()))]) # model for classification

    predict_labels(classifier, X_train, y_train, X_test, y_test, target_names)
    evaluate_model(classifier, X_train, y_train, X_test, y_test, target_names)

main()
