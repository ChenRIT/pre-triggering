# A pre-trigger that identifies information-seeking questions

import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

import spacy
import benepar
from benepar.spacy_plugin import BeneparComponent

import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

nlp = spacy.load('en')
nlp.add_pipe(BeneparComponent('benepar_en'))

path_to_vectorizer = './models/vectorizer_0.pkl'
path_to_model = './models/logi_model_0.pkl'

def timeit(func):
    def timed(*args, **kw):
        ts = time.time()
        res = func(*args, **kw)
        te = time.time()
        diff = te - ts
        logging.debug("Time for {}: {}.\n".format(func.__name__, diff))

        return res

    return timed

class pre_trigger():
    """
    A pre_trigger that (1) identify questions, and
    (2) identify information seeking questions
    """
    def __init__(self, model='logistic_regression'):
        if model == 'logistic_regression':
            self.vectorizer = joblib.load(path_to_vectorizer)
            self.logi_model = joblib.load(path_to_model)
            # Parse a test sentence:
            self.identify_info("Good morning.")

    @timeit        
    def identify_question(self, sent):
        """
        Identify if a sentence is a question or not.
        """
        doc = nlp(sent)
        parse_sent = list(doc.sents)[0]
        
        if 'SQ' in parse_sent._.parse_string or \
           'SBARQ' in parse_sent._.parse_string:
            return True
        else:
            return False

    @timeit    
    def _clean_sent(self, sent):
        """
        Clean a sentence by removing non-English words and numbers.
        """
        cleaned_sent = re.sub("[^a-zA-Z]", " ", sent)
        lower_sent = cleaned_sent.lower()
        new_sent = ' '.join(lower_sent.split())

        logging.debug('Cleaned sent: {}'.format(new_sent))
        return new_sent
        
    @timeit
    def identify_info(self, sent):
        """
        Determine if a sentence is a information-seeking sentence or not.
        """
        # Predict whether the sentence is about information seeking or not.
        clean_sent = self._clean_sent(sent)

        # Use constituency parsing tree to determine if the sentence is a question
        is_question = self.identify_question(clean_sent)
        if not is_question:
            return (is_question, False)

        # Use logistic regression to determine if the sentence
        # is about information-seeking.
        features_sent = self.vectorizer.transform([clean_sent])
        prediction = self.logi_model.predict(features_sent)
        
        if prediction[0] == 1:
            return (is_question, True)
        else:
            return (is_question, False)



def test():
    sent1 = "When is the pool closed?"
    sent2 = "Can you deliver a towel to our room?"
    sent3 = "I like the burger in your restaurant."
    
    pre_trigger = pre_trigger()
    
    sent1_res = pre_trigger.identify_info(sent1)
    print("sent1's result: {}".format(sent1_res))

    sent2_res = pre_trigger.identify_info(sent2)
    print("sent2's result: {}".format(sent2_res))
    
    sent3_res = pre_trigger.identify_info(sent3)
    print("sent3's result: {}".format(sent3_res))

        
if __name__ == '__main__':
    sent1 = "If I needed to cancel- what time do I have to do that by? 4?"
    sent2 = "Good. 2 questions, is there a fitness center and where is a good place to eat (for 1)"    

    pre_trigger = pre_trigger()

    sent1_res = pre_trigger.identify_info(sent1)
    sent2_res = pre_trigger.identify_info(sent2)

    print("sent1's result: {}".format(sent1_res))
    print("sent2's result: {}".format(sent2_res))    



    
