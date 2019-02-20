import loader
from senser import Senser
from nltk.corpus import wordnet as wn

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:15:31 2019

@author: jjnun
"""
# Quick helper function to return whether an item is in a list, 1 = true, 0 equals false
def inList(an_item, a_list):
    if (an_item in a_list):
        return 1
    else:
        return 0

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
lesks_acc = 0 # Accuracy accumulator for choosing sense with Lesk's algorithm
n_instances = len(dev_instances)
##print(type(dev_instances.values()))

for instance_id, instance in dev_instances.items():
    # Retrieve correct sense
    ##print(type(instance))
    true_sense = dev_key[instance_id]
    # Find predicted senses
    senser = Senser(instance)
    base_pred  = senser.predictBase()
    lesks_pred = senser.predictLesks()
    ##print(base_pred)
    ##print(true_sense)
    base_acc  += inList(base_pred, true_sense)
    lesks_acc += inList(lesks_pred, true_sense)


# Calculate and print accuracies
base_acc = base_acc/n_instances
lesks_acc = lesks_acc/n_instances

print("Base accuracy is: " + str(base_acc))
print("Lesk's algorithm accuracy is: " + str(lesks_acc))


# Debugging

#print(dev_instances)

#print(dev_instances['d001.s029.t001'].getID())

