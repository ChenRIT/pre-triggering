# Test a logistic regression model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import re
#import pickle
from sklearn.externals import joblib

test = pd.read_csv('./test_0.csv',
                      index_col=0,
                      skip_blank_lines=True)

vectorizer = joblib.load('vectorizer_0.pkl')

# Evaluate the model on the testing dataset
features_test = vectorizer.transform(test['cleaned_sent'].tolist())
#print("Dimension of the testing data: {}".format(features_test.shape))

logi_model = joblib.load('logi_model_0.pkl')
predictions = logi_model.predict(features_test)
print('Accuracy results: {}'.format(metrics.accuracy_score(test['label_ml'],
                                                           predictions)))
print('Precision results: {}'.format(metrics.precision_score(test['label_ml'],
                                                             predictions)))
print('Recall results: {}'.format(metrics.recall_score(test['label_ml'],
                                                       predictions)))
print('F1 results: {}'.format(metrics.f1_score(test['label_ml'],
                                                   predictions)))
