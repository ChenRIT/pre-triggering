# Train a logistic regression model for information seeking questions
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
    sent = row['would_you_be_able_to_tell_me_what_the_balance_on_my_room_is']
    cleaned_sent = re.sub("[^a-zA-Z]", " ", sent)
    lower_sent = cleaned_sent.lower()
    new_sent = ' '.join(lower_sent.split())

    return new_sent

# Load the training and testing dataset
train_df = pd.read_csv('./dataset/hotel_answers_train_3600.csv',
                       index_col=0,
                       skip_blank_lines=True)

test_df = pd.read_csv('./dataset/hotel_answers_test_400.csv',
                      index_col=0,
                      skip_blank_lines=True)

# Clean sentences by removing non-English words and numbers
train_df.loc[:, 'cleaned_sent'] = train_df.apply(sent_clean, axis=1)
test_df.loc[:, 'cleaned_sent'] = test_df.apply(sent_clean, axis=1)

datasets = []

# Training and testing datasets without junk sentences
train_df_no_junk = train_df.copy()[train_df['label_ml'] != 2]
test_df_no_junk = test_df.copy()[test_df['label_ml'] != 2]
datasets.append((train_df_no_junk, test_df_no_junk))


# Training and testing datasets with junk sentences
train_df_with_junk = train_df
train_df_with_junk.loc[train_df_with_junk['label_ml'] == 2, 'label_ml'] = 0
# print("train_df_with_junk:\n")
# print(train_df_with_junk.head(10))
test_df_with_junk = test_df
test_df_with_junk.loc[test_df_with_junk['label_ml'] == 2, 'label_ml'] = 0
datasets.append((train_df_with_junk, test_df_with_junk))
# print("test_df_with_junk:\n")
# print(test_df_with_junk.head(10))


for i, (train, test) in enumerate(datasets):
    print("\n--------------\n")
    # Feature engineering
    vectorizer = CountVectorizer()
    features_train = vectorizer.fit_transform(train['cleaned_sent'].tolist())
    joblib.dump(vectorizer, './pre-trigger/models/vectorizer_'+str(i)+'.pkl')
    print("Dimension of the training data: {}".format(features_train.shape))

    # Train a logistic regression model on the training dataset
    logi_model = LogisticRegression()
    logi_model.fit(features_train, train['label_ml'])
    #pickle.dump(logi_model, open(('logi_model_' + str(i)), 'wb'))
    joblib.dump(logi_model, './pre-trigger/models/logi_model_'+str(i)+'.pkl')
    print('Training results: {}'.format(logi_model.score(features_train,
                                                         train['label_ml'])))

    # Evaluate the model on the testing dataset
    features_test = vectorizer.transform(test['cleaned_sent'].tolist())
    #print("Dimension of the testing data: {}".format(features_test.shape))

    predictions = logi_model.predict(features_test)
    test.loc[:, 'prediction'] = predictions
    print('Accuracy results: {}'.format(metrics.accuracy_score(test['label_ml'],
                                                               predictions)))
    print('Precision results: {}'.format(metrics.precision_score(test['label_ml'],
                                                                 predictions)))
    print('Recall results: {}'.format(metrics.recall_score(test['label_ml'],
                                                           predictions)))
    print('F1 results: {}'.format(metrics.f1_score(test['label_ml'],
                                                   predictions)))

    
    # Print the train and test datasets
    train.to_csv('./train/train_{}.csv'.format(i))
    test.to_csv('./train/test_{}.csv'.format(i))

    print("\n--------------\n")    
