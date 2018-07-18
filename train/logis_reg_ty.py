# Train a logistic regression model for information seeking questions

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import re


def label(row):
    user_label = row['choose_the_most_natural_answer_to_the_question_above']
    if user_label == 2:
        return 1
    else:
        return 0


def sent_clean(row):
    sent = row['would_you_be_able_to_tell_me_what_the_balance_on_my_room_is']
    cleaned_sent = re.sub("[^a-zA-Z]", " ", sent)
    lower_sent = cleaned_sent.lower()
    new_sent = ' '.join(lower_sent.split())

    return new_sent


# Load dataset
df = pd.read_csv("./dataset/test_hotel_answer_aggregate_200.csv")

# Create label
df.loc[:, 'label'] = df.apply(label, axis=1)

# Clean sentences by removing non-English words and numbers
df.loc[:, 'cleaned_sent'] = df.apply(sent_clean, axis=1)

# Create the training and testing dataset
df_size = len(df)
train_size = int(0.9 * df_size)

train_df = df.iloc[0:train_size]
test_df = df.iloc[train_size:df_size]

# Feature engineering
vectorizer = CountVectorizer()
features_train = vectorizer.fit_transform(train_df['cleaned_sent'].tolist())

# print("Dimension of the training data: {}".format(features_train.shape))

# Train a logistic regression model on the training dataset
logi_model = LogisticRegression()
logi_model.fit(features_train, train_df['label'])
print('Training results: {}'.format(logi_model.score(features_train,
                                                     train_df['label'])))

# Evaluate the model on the testing dataset
features_test = vectorizer.transform(test_df['cleaned_sent'].tolist())
print("Dimension of the testing data: {}".format(features_test.shape))

predictions = logi_model.predict(features_test)
test_df.loc[:, 'prediction'] = predictions
print('Testing results: {}'.format(metrics.accuracy_score(test_df['label'],
                                                          predictions)))

# Print the train and test datasets
train_df.to_csv('train.csv')
test_df.to_csv('test.csv')
