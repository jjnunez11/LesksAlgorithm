from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:17:34 2019

@author: jjnun
"""

# Set whether to be verbose for a debug mode
## debug_on = False
debug_on = False

class Senser:
    def __init__(self, wsdi):
        self.wsdi = wsdi        
        lemma = wsdi.getLemma().decode('UTF-8').lower()
        self.lemma = lemma
        
        # Prepare shared resources
        
        # Make list of words to remove
        ignore_words = list(string.punctuation) + stopwords.words('english') # No punctation or stop words
        ignore_words.append('@card@') # Remove this card marker
        ignore_words.append(lemma) # Remove Lemma, improves performancae
        ignore_words.append('--') # Remove -- which sometimes appears
        ignore_words.append('\'s') # Remove this posession marker
        self.ignore_words = ignore_words
        
        # Instantiate a lemmatizer
        lemmatizer = WordNetLemmatizer()
        self.lemmatizer = lemmatizer
        
        # Get senses
        self.senses = wn.synsets(lemma, pos = wn.NOUN)
        
        # Process context
        raw_context = self.wsdi.getContext()
        context = process_context(raw_context, lemma, ignore_words, lemmatizer)
        self.context = context
        
    def predictBase(self):
        # Convert byte to str
        lemma = self.lemma
        # Return the sense (keys) for the most probable (top) sense of this lemma
        base_pred = []
        
        #lesks_pred = sense.lemmas()[0].key()
        for l in wn.synsets(lemma)[0].lemmas(): # This is the top sense
            # Include all the keys of this top sense
            base_pred.append(l.key()) 
            
        return base_pred
        
    
#    # Simplified Lesk's Algorithm, where if all senses have 0 overlap, chooses first
#    def predict_slesks_tkft(self):        
#        if debug_on: print('This is the lemma: ' + str(self.lemma))
#        
#        best_overlap = -1
#        for sense in self.senses:
#            
#            # Process sense's definition
#            raw_definition = sense.definition()
#            definition = process_definition(raw_definition, self.lemma, self.ignore_words, self.lemmatizer)    
#            
#            # Compute a score for each sense per simplified Lesk algorithm
#            overlap = len(set(self.context) & set(definition))
#            if debug_on: print('This defs overlap: ' + str(overlap) + '\n')
#            
#            # Store if best            
#            if (overlap > best_overlap):
#                best_overlap = overlap
#                slesks_tkft_pred = []
#                #lesks_pred = sense.lemmas()[0].key()
#                for l in sense.lemmas():
#                    slesks_tkft_pred.append(l.key())
#                ##print('here is first lemma.key: ' + str(sense.lemmas()[0].key()))
#                ##print('here is one lemma.key: ' + str(sense.lemmas()[1].key()))
#                
#        return slesks_tkft_pred
    
    # Simplified Lesk's Algorithm, making no prediction if all senses have 0 overlap
    def predict_slesks(self, default_to_first=False):
        if debug_on: print('This is the lemma: ' + str(self.lemma))
        slesks_pred = []
        
        # If default_to_first is True, if no overlap anywhere, predict first sense, otherwise do not
        if default_to_first:
            best_overlap = -1
        else:
            best_overlap = 0
        
        for sense in self.senses:
            
            # Process sense's definition
            raw_definition = sense.definition()
            definition = process_definition(raw_definition, self.lemma, self.ignore_words, self.lemmatizer)    
            
            # Compute a score for each sense per simplified Lesk algorithm
            overlap = len(set(self.context) & set(definition))
            if debug_on: print('This defs overlap: ' + str(overlap) + '\n')
            
            # Store if best, but only if it's above zero, can break tie 
            if (overlap > best_overlap):
                best_overlap = overlap
                slesks_pred = []
                #lesks_pred = sense.lemmas()[0].key()
                for l in sense.lemmas():
                    slesks_pred.append(l.key())
                ##print('here is first lemma.key: ' + str(sense.lemmas()[0].key()))
                ##print('here is one lemma.key: ' + str(sense.lemmas()[1].key()))
                
        return slesks_pred
    
    # Original Lesk's Algorithm, making no prediction if all senses have 0 overlap
    def predict_olesks(self):
        if debug_on: print('This is the lemma: ' + str(self.lemma))
        ignore_words = self.ignore_words
        lemmatizer = self.lemmatizer
        olesks_pred = []
        
        best_overlap = 0 # -1 will choose first sense even if none have matches, 0 will ignore unless actual match
        for sense in self.senses:
            
            # Process sense's definition
            raw_definition = sense.definition()
            definition = process_definition(raw_definition, self.lemma, ignore_words, lemmatizer)    
            
            # Compute a score for each sense per original Lesk algorithm
            overlap = olesk_overlap(ignore_words, lemmatizer, self.context, definition)
            if debug_on: print('This defs overlap: ' + str(overlap) + '\n')
            
            # Store if best, but only if it's above zero, can break tie 
            
            if (overlap > best_overlap):
                best_overlap = overlap
                olesks_pred = []
                #lesks_pred = sense.lemmas()[0].key()
                for l in sense.lemmas():
                    olesks_pred.append(l.key())
                ##print('here is first lemma.key: ' + str(sense.lemmas()[0].key()))
                ##print('here is one lemma.key: ' + str(sense.lemmas()[1].key()))
                
        return olesks_pred

# Returns the number of overlap between the target's definition, and the definition of all of the context words
def olesk_overlap(ignore_words, lemmatizer, context, target_definition):
    # Generate list of all of the context word's definitions, processed as our definitions always are
    context_definitions = []
    for s in context:
        s_senses = wn.synsets(s, pos = wn.NOUN)
        for sense in s_senses:
            context_definition = process_definition(sense.definition(), [], ignore_words, lemmatizer)
            context_definitions = context_definitions + context_definition
    
    overlap = len(set(target_definition) & set(context_definitions))
    
    return overlap

def process_context(raw_context, lemma, ignore_words, lemmatizer):
    
    #Process context from array of bytes into arrays of appropriate strings
    str_context = [b.decode('utf-8') for b in raw_context]
    # Convert to lower case
    tokens_context = [s.lower() for s in str_context]
    # Remove punctuation and stop words
    context = [t for t in tokens_context if not t in ignore_words]
    # Convert underscored or hyphenated words to words with a space
    context = [re.sub(r'([a-zA-Z]+)[_|-]([a-zA-Z]+)',r'\1 \2',t) for t in context]
    # Lemmatize
    lem_context = [lemmatizer.lemmatize(t) for t in context]
    
    ## Debugging
    if debug_on:
        print('Raw context: ' + str(raw_context))
        print('Clean context: ' + str(context) + '\n')
      
    return lem_context
    
def process_definition(raw_definition, lemma, ignore_words, lemmatizer):
    
    #Process sense's definition
    tokens_def = word_tokenize(raw_definition)
    # Convert to lower case
    tokens_def = [s.lower() for s in tokens_def]
    # Remove punctuation and stop words
    definition = [t for t in tokens_def if not t in ignore_words]
    # Lemmaziation
    lem_definiiton = [lemmatizer.lemmatize(t) for t in definition]
    
    ## Debugging
    if debug_on:
        print('Raw def: ' + str(raw_definition))
        print('Clean def: ' + str(definition))
    
    return lem_definiiton

    

#policy%1:09:00::
##print(wn.synsets('policy'))
##print(wn.synsets('policy')[0].lemmas())#[0].key())