# Identify questions based on constituency tags

import spacy
import benepar
from benepar.spacy_plugin import BeneparComponent

nlp = spacy.load('en')
nlp.add_pipe(BeneparComponent('benepar_en'))

sent1 = u"Why have I achieved so much?"
sent2 = u"I have achieved quite a lot."
sent3 = u'Is this true?'

doc1 = nlp(sent1)
sent1 = list(doc1.sents)[0]
print(sent1._.parse_string)
if 'SQ' in sent1._.parse_string or \
   'SBARQ' in sent1._.parse_string:
    print("sent1 is a question.")
else:
    print("sent1 is not a question.")

doc2 = nlp(sent2)
sent2 = list(doc2.sents)[0]
print(sent2._.parse_string)
if 'SQ' in sent2._.parse_string or \
   'SBARQ' in sent2._.parse_string:
    print("sent2 is a question.")
else:
    print("sent2 is not a question.")

doc3 = nlp(sent3)
sent3 = list(doc3.sents)[0]
print(sent3._.parse_string)
if 'SQ' in sent3._.parse_string or \
   'SBARQ' in sent3._.parse_string:
    print("sent3 is a question.")
else:
    print("sent3 is not a question.")
    
