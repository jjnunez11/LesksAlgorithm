# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:17:34 2019

@author: jjnun
"""

class Senser:
    def __init__(self, wsdi):
        self.wsdi = wsdi        # id of the WSD instance
        # self.lemma = lemma      # lemma of the word whose sense is to be resolved
        #self.context = context  # lemma of all the words in the sentential context
        # self.index = index      # index of lemma within the context
        
    def predictBase(self):
        return 1
    
    def predictLesks(self):
        return 1