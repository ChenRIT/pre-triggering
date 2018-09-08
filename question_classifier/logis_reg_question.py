# Train a logistic regression model for question classifications
# Two datasets: (train_with_no_junk, test_with_no_junk), (train_with_junk, test_with_junk)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import numpy as np
import re
#import pickle
from sklearn.externals import joblib

def sent_clean(row):
    sent = row['text']
    cleaned_sent = re.sub("[^a-zA-Z]", " ", sent)
    lower_sent = cleaned_sent.lower()
    new_sent = ' '.join(lower_sent.split())

    return new_sent

# Load the training and testing dataset
pos_df = pd.read_csv('./dataset/questions.csv',
                       skip_blank_lines=True)
neg_df = pd.read_csv('./dataset/non_questions.csv',
                           skip_blank_lines=True)

pos_df.loc[:, 'label_ml'] = 1
neg_df.loc[:, 'label_ml'] = 0

data_df = pd.concat([pos_df[['would_you_be_able_to_tell_me_what_the_balance_on_my_room_is', 'label_ml']].rename(columns={'would_you_be_able_to_tell_me_what_the_balance_on_my_room_is': 'text'}), neg_df[['text', 'label_ml']]])

data_df = data_df.sample(frac=1).reset_index(drop=True)
train_size = int(len(data_df) * 9 / 10)

print(data_df.head())

train_df = data_df.iloc[:train_size]
test_df = data_df.iloc[train_size:]

# Clean sentences by removing non-English words and numbers
train_df.loc[:, 'cleaned_sent'] = train_df.apply(sent_clean, axis=1)
test_df.loc[:, 'cleaned_sent'] = test_df.apply(sent_clean, axis=1)

print("\n--------------\n")
# Feature engineering
vectorizer = CountVectorizer()
features_train = vectorizer.fit_transform(train_df['cleaned_sent'].tolist())
joblib.dump(vectorizer, './vectorizer_questions.pkl')
print("Dimension of the training data: {}".format(features_train.shape))

# Train a logistic regression model on the training dataset
logi_model = LogisticRegression()
logi_model.fit(features_train, train_df['label_ml'])
#pickle.dump(logi_model, open(('logi_model_' + str(i)), 'wb'))
joblib.dump(logi_model, './logi_model_questions.pkl')
print('Training results: {}'.format(logi_model.score(features_train,
                                                     train_df['label_ml'])))

# Evaluate the model on the testing dataset
features_test = vectorizer.transform(test_df['cleaned_sent'].tolist())
#print("Dimension of the testing data: {}".format(features_test.shape))

predictions = logi_model.predict(features_test)
test_df.loc[:, 'prediction'] = predictions
print('Accuracy results: {}'.format(metrics.accuracy_score(test_df['label_ml'],
                                                           predictions)))
print('Precision results: {}'.format(metrics.precision_score(test_df['label_ml'],
                                                             predictions)))
print('Recall results: {}'.format(metrics.recall_score(test_df['label_ml'],
                                                       predictions)))
print('F1 results: {}'.format(metrics.f1_score(test_df['label_ml'],
                                               predictions)))


# Print the train and test datasets
#train.to_csv('./train/train_{}.csv'.format(i))
#test.to_csv('./train/test_{}.csv'.format(i))

print("\n--------------\n")    
