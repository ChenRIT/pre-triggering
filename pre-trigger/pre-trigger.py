# A pre-trigger that identifies information-seeking questions

import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

# import spacy
# import benepar
# from benepar.spacy_plugin import BeneparComponent

import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

sents = [
    r"A what time exactly I leaving tomorrow",
    r"Also massages how much and available times?",
    r"Also when we check out?",
    r"Also, she is getting in earlier in the day so can we request an early check out?",
    r"Any guidance about where to park?",
    r"Any idea what time my room will be ready for checkin?",
    r"Any way you can tell me where the closest rental car company is from the Corazon?",
    r"Check in is what time?",
    r"Enjoyable stay, what time is check out tomorrow?",
    r"Enjoying our stay (: do you guys have car services?",
    r"Good morning, we are wondering how we get into the pool and gym area?",
    r"Great I will be looking to set up a massage at the spa is there a simple way for me to do that?",
    r"Great, what is the spa schedule",
    r"Hello Pfister Bulter, can you confirm check out please?",
    r"Hello Shelly, what about the parking?",
    r"Hello where my supposed to park?",
    r"Hello, we are planning to be there in half hour or so, do you think our room is going to be ready?",
    r"Hey i am mrs vazquez in room 1020 what time the store close",
    r"Hey what his the earliest I can do check in ?",
    r"Hey, I'll be driving in from Arkansas, was wondering what's the earliest check in time available?",
    r"Hi - we land at 11:45 am, what time is check-in?",
    r"Hi I was wondering what's the earliest check in time?",
    r"Hi There - curios to know what a late check out would cost us?",
    r"Hi again, I was wondering what the wifi password is?",
    r"Hi there - just wondering how to check in for a stay this evening.",
    r"Hi what is the address of the hotel",
    r"Hi what is the username and password for WiFi",
    r"Hi, I was just wondering what time would be the earliest we could check in today?",
    r"Hi, just about to check out and wondering where we could leave bags?",
    r"Hi,.Just checking what time check in is?",
    r"How early can we check in tomorrow, I'm driving about 3 hours away",
    r"I am aware that the check in time is at 3pm, is there any way we could check in earlier?",
    r"I can't find directions for how to check in.",
    r"I do plan on bringing a car, do you need anything from me for that?",
    r"I have a question can we leave at 12pm?",
    r"I have the password, what my username for WiFi?",
    r"I need a message do you have a spa?",
    r"I wanted to check what time is check out",
    r"I was wondering what the username and password is for the wifi.",
    r"I was wondering what time we can check in.",
    r"I'm arriving on Wednesday and was wondering what time I can check in. .Thank",
    r"I'm curious how I check in during an upcoming stay.",
    r"I'm staying in room 212 and was wondering what time check out is?",
    r"I'm still confused on how to check in",
    r"I'm wondering what time I can check in today?",
    r"Just curious of what we will need to do tomorrow to check in.",
    r"Just curious what time I could check in early check-in is not necessary",
    r"Just curious what time we can check in",
    r"Just wondering what time check in is??",
    r"Looking forward to our stay as well- wondering what time is check in?",
    r"Meant to say what do we need to do for check in",
    r"Morning, what time is checkout.",
    r"My room is booked till tomorrow, what time is check out usually?",
    r"Oh one more thing...how close is your nearest grocery store or Walmart?",
    r"One quick question is there wifi access?",
    r"Or flights at 3 is there any way we could get late check-out.",
    r"Our children would like to know what time the pool opens..",
    r"Palmer and I was wondering what time is the arrival time tomorrow?",
    r"Pool opens at what time?",
    r"Question: what time is check out",
    r"Remind me what time check in is?",
    r"Rm 244 what time does the indoor pool and outdoor hot tub close tonight",
    r"Room 604 - have wifi code what is user name keeps telling me case sensitivity",
    r"Sorry to bother Ben...I don't understand how to get in the gym.",
    r"Stupid question, but what time is check-in?",
    r"Thank you ,what time is check in and out.",
    r"Thank you, what time the pools close?",
    r"Thank you--we are arriving early what is the earliest we can check in?",
    r"Thanks so much what time is check in",
    r"Thanks what time does spa open",
    r"The hotel address I have is not working in my GPS is there another address that might work",
    r"This is josh from room 1050 what time is check out tomorrow",
    r"This is mrs.marquez room 1024 .what time do the pool closes?",
    r"We are in room 322 and we were wondering what the wifi username is",
    r"We are interested in early check in and late check out, can you tell me the availability?",
    r"We are now arriving at around 12:45 - 1:00 how is early check in looking??",
    r"We are parked, which says three hour parking, is this garage okay or do we need to move?",
    r"We are wondering when we can check in today?",
    r"We did wonder how soon we would be able to check in?",
    r"We need a tutorial on how to keep our hotel wifi working.",
    r"We were wondering what hours the pool is open?",
    r"We will be arriving early in the morning, can you tell me about early check in",
    r"What is wifi user name, I think what she gave me is the password right?",
    r"What time is check out and do you offer late check-out",
    r"Wondering how to make an appt for a massage in the well spa",
    r"Yes I would like to know what time is check out",
    r"and I was just wondering how to check in to the apartment",
    r"but not sure where I park?",
    r"good morning..what time is checkout?..thanka",
    r"we are in town just having lunch, can we check in any time?",
    r"what is check in time",
    r"what time is The Breakfast and how much",
    r"what time is check in and yes we just got married"
]

# sents = [
#     r"today's a beautiful day",
#     r"The sky is blue",
#     r"We went to the beach.",
#     r"I just arrived.",
#     r"Tell me which floor am I on?"
# ]

# nlp = spacy.load('en')
# nlp.add_pipe(BeneparComponent('benepar_en'))

path_to_vectorizer = './models/vectorizer_0.pkl'
path_to_model = './models/logi_model_0.pkl'
path_to_qa_vectorizer = './models/vectorizer_questions.pkl'
path_to_qa_model = './models/logi_model_questions.pkl'


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
            self.qa_vectorizer = joblib.load(path_to_qa_vectorizer)
            self.qa_logi_model = joblib.load(path_to_qa_model)            
            self.vectorizer = joblib.load(path_to_vectorizer)
            self.logi_model = joblib.load(path_to_model)
            # Parse a test sentence:
            self.identify_info("Good morning.")

    #@timeit        
    # def identify_question_spacy(self, sent):
    #     """
    #     Identify if a sentence is a question or not using SpaCy.
    #     """
    #     doc = nlp(sent)
    #     parse_sent = list(doc.sents)[0]
        
    #     if 'SQ' in parse_sent._.parse_string or \
    #        'SBARQ' in parse_sent._.parse_string:
    #         return True
    #     else:
    #         return False

    def identify_question(self, sent):
        """
        Identify if a sentence is a question or not using pre-trained question classifier.
        """
        features_sent = self.qa_vectorizer.transform([sent])
        prediction = self.qa_logi_model.predict(features_sent)
        
        if prediction[0] == 1:
            return True
        else:
            return False
    

    #@timeit    
    def _clean_sent(self, sent):
        """
        Clean a sentence by removing non-English words and numbers.
        """
        cleaned_sent = re.sub("[^a-zA-Z\?]", " ", sent)
        lower_sent = cleaned_sent.lower()
        new_sent = ' '.join(lower_sent.split())

        logging.debug('Cleaned sent: {}'.format(new_sent))
        return new_sent
        
    #@timeit
    def identify_info(self, sent):
        """
        Determine if a sentence is a information-seeking sentence or not.
        """
        # Predict whether the sentence is about information seeking or not.
        clean_sent = self._clean_sent(sent)

        # Use constituency parsing tree to determine if the sentence is a question
        is_question = self.identify_question(sent)        
        # if not is_question:
        #     return (is_question, False)

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
    # sent1 = "Hi I'm interested in booking a room for sxsw week, what dates are available between March 9-18?"
    # pre_trigger = pre_trigger()
    # sent1_res = pre_trigger.identify_info(sent1)
    # print("sent1's result: {}".format(sent1_res))

    pre_trigger = pre_trigger()
    results = []
    for sent in sents:
        sent_res = pre_trigger.identify_info(sent)
        results.append((sent, sent_res))

    for res in results:
        print("sent: {} \n result: {}".format(res[0], res[1]))



    
