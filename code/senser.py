from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import string
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:17:34 2019

@author: jjnun
"""

class Senser:
    def __init__(self, wsdi):
        self.wsdi = wsdi        
        
        ##print(type(wn.get_version()))
        
        #Ensure WordNet version 3.0
        ##if (float(wn.get_version()) != 3.0):
        ##    raise ValueError('WordNet version is not 3.0, instead is: ' + wn.get_version())
        

        
    def predictBase(self):
        # Convert byte to str
        lemma = self.wsdi.getLemma().decode('UTF-8')
        # Return the sense (keys) for the most probable (top) sense of this lemma
        return wn.synsets(lemma)[0].lemmas()[0].key()
    
    def predictLesks(self):
        lemma = self.wsdi.getLemma().decode('UTF-8')
        senses = wn.synsets(lemma, pos = wn.NOUN)
        
        best_score = -1
        for sense in senses:
            # Compute a score for each sense per simplified Lesk algorithm
            score = overlap(sense, self.wsdi.getContext())
            
            if (score > best_score):
                best_score = score
                lesks_pred = sense.lemmas()[0].key()
            
            ## TO DO REMOVE
        
        return lesks_pred
    
def overlap(sense, raw_context):
    # We wont count punctuation or stop words
    stop_punc_words = stopwords.words('english') + list(string.punctuation)
    
    #Process context from array of bytes into arrays of appropriate strings
    str_context = [b.decode('utf-8') for b in raw_context]
    # Convert to lower case
    tokens_context = [s.lower() for s in str_context]
    # Remove punctuation and stop words
    context = [t for t in tokens_context if not t in stop_punc_words] 
    
    #Process sense's definition
    raw_def = sense.definition()
    tokens_def = word_tokenize(raw_def)
    # Convert to lower case
    tokens_def = [s.lower() for s in tokens_def]
    # Remove punctuation and stop words
    definition = [t for t in tokens_def if not t in stop_punc_words]
    
    # Calculate score as similiarity 
    overlap_score = len(set(context) & set(definition))
    
    
    print('here is the sense definition')
    print(definition)
    print('here is the context:')
    print(context)
    print('Score: ' + str(overlap_score))
    
    
    return overlap_score    

#policy%1:09:00::
##print(wn.synsets('policy'))
##print(wn.synsets('policy')[0].lemmas())#[0].key())