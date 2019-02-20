from nltk.corpus import wordnet as wn
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
        return 1