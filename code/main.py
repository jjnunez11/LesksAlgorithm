import loader
from senser import Senser
from nltk.corpus import wordnet as wn

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:15:31 2019

@author: jjnun
"""
# Quick helper function to return whether an item is in a list, 1 = true, 0 equals false
def mutual_element(a_list, b_list):
    a_set = set(a_list)
    b_set = set(b_list)
    if len(a_set.intersection(b_set)) > 0:
        return 1
    else:
        return 0
    
# Quick helper to calculate F1    
def my_F1(prec, recall):
    return 2*(prec * recall)/(prec + recall)



# Load with given code
data_f = 'multilingual-all-words.en.xml'
key_f = 'wordnet.en.key'
dev_instances, test_instances = loader.load_instances(data_f)
dev_key, test_key = loader.load_key(key_f)
    
# IMPORTANT: keys contain fewer entries than the instances; need to remove them
dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}

#Calculate accuracies
base_acc  = 0 # Accuracy accumulator for choosing sense to be most common (#1 in wordsense)
slesks_tkft_acc = 0 # Accuracy accumulator for choosing sense with simplified lesks taking first if all 0
slesks_acc = 0 # Accuracy accumulator for choosing sense with simplified lesks without using 0
slesks_n_rvd = 0 # Number retrieved by simplified lesks, for prec/recall


n_instances = len(dev_instances)


##print(type(dev_instances.values()))

##TODO REMOVE DEBUGGING
debug = 0





for instance_id, instance in dev_instances.items():
    # Retrieve correct sense
    ##print(type(instance))
    true_sense = dev_key[instance_id]
    # Find predicted senses
    senser = Senser(instance)
    base_pred  = senser.predictBase()
    base_acc  += mutual_element(base_pred, true_sense)
    ##print(instance.getContext())
    
#    ## Debgugging
#    if debug == 176:
#        slesks_tkft_pred = senser.predictLesks()
#        print('Predicted senses: ' + str(slesks_tkft_pred))
#        print(wn.lemma_from_key(slesks_tkft_pred[0]).synset().definition())
#        print('\n')
#        print('The true senses are: ' + str(true_sense))
#        print(wn.lemma_from_key(true_sense[0]).synset().definition())
#        print('\n')
#        print('True?: ' + str(mutual_element(slesks_tkft_pred, true_sense)))
#    slesks_tkft_acc = 0 ## TODO REMOVE
#    debug += 1 ## TODO REMOVE
    
    ## Debugging 2
#    slesks_tkft_pred = senser.predictLesks()
#    if mutual_element(base_pred, true_sense) > mutual_element(slesks_tkft_pred, true_sense):
#        print(' ' + str(debug) + ' ')
#    slesks_tkft_acc = 0
#    debug += 1
#    

    
    ## Actual
    # Simplified Lesks algorithm that breaks a tie, including 0, by taking first sense
    slesks_tkft_pred = senser.predict_slesks_tkft()
    slesks_tkft_acc += mutual_element(slesks_tkft_pred, true_sense)
    
    # Simplified Lesks that only predicts if score > 0, will break tie by first in this case
    slesks_pred = senser.predict_slesks()
    slesks_acc += mutual_element(slesks_pred, true_sense)
    # Increment number retrived if it found something
    if len(slesks_pred) > 0: slesks_n_rvd += 1 

# Calculate and print accuracies, precision, recall
base_acc = base_acc/n_instances
slesks_tkft_acc = slesks_tkft_acc/n_instances
slesks_prec = slesks_acc/slesks_n_rvd
slesks_acc = slesks_acc/n_instances
slesks_f1 = my_F1(slesks_prec, slesks_acc)


print("Total examples: " + str(debug))
print("Base accuracy is: " + str(base_acc))
print("Simplified Lesk's algorithm, breaking 0 by first sense, accuracy/recall/precision is: " + str(slesks_tkft_acc))
print("Simplified Lesk's algorithm, breaking 0 by first sense, F1 is: " + str(my_F1(slesks_tkft_acc, slesks_tkft_acc)))
print("Simplified Lesk's algorithm, ignoring 0's, accuracy/recall is: " + str(slesks_acc))
print("Simplified Lesk's algorithm, ignoring 0's, precision is: " + str(slesks_prec))
print("Simplified Lesk's algorithm, ignoring 0's, F1 is: " + str(slesks_f1))

# Debugging

#print(dev_instances)

#print(dev_instances['d001.s029.t001'].getID())


