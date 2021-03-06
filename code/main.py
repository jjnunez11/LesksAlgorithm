import loader
from senser import Senser
from nltk.corpus import wordnet as wn

"""
Created on Tue Feb 19 16:15:31 2019

@author: jjnun
"""
# Quick helper function to return whether an item is both list, 1 = true, 0 equals false
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
olesks_tkft_acc = 0 # Accuracy accumulator for choosing sense with original lesks, takin first if all 0
olesks_acc = 0 # Accuracy accumulator for choosing sense with original lesks without using 0s
olesks_n_rvd = 0 # Number retrieved by original lesks, for prec/recall

n_instances = len(dev_instances)

for instance_id, instance in dev_instances.items():
    # Retrieve correct sense
    true_sense = dev_key[instance_id]
    # Find predicted senses
    senser = Senser(instance)
    base_pred  = senser.predictBase()
    base_acc  += mutual_element(base_pred, true_sense)
    
    # Simplified Lesks that only predicts if score > 0, will break tie by first in this case
    slesks_pred = senser.predict_slesks(default_to_first=False)
    slesks_acc += mutual_element(slesks_pred, true_sense)
    # Increment number retrived if it found something
    if len(slesks_pred) > 0: slesks_n_rvd += 1
    
    # Simplified Lesks algorithm that breaks a tie, including 0, by taking first sense
    slesks_tkft_pred = senser.predict_slesks(default_to_first=True)
    slesks_tkft_acc += mutual_element(slesks_tkft_pred, true_sense)
    
    # Original Lesks algorithm that only predicts if score > 0, breaks tie by first in this case
    olesks_pred = senser.predict_olesks(default_to_first=False)
    olesks_acc += mutual_element(olesks_pred, true_sense)
    # Increment number retrived if it found something
    if len(olesks_pred) > 0: olesks_n_rvd += 1

    # Original Lesks algorithm that predicts the first sense if no matches found
    olesks_tkft_pred = senser.predict_olesks(default_to_first=True)
    olesks_tkft_acc += mutual_element(olesks_tkft_pred, true_sense)      

# Calculate and print accuracies, precision, recall
base_acc = base_acc/n_instances

slesks_tkft_acc = slesks_tkft_acc/n_instances
slesks_prec = slesks_acc/slesks_n_rvd
slesks_acc = slesks_acc/n_instances
slesks_f1 = my_F1(slesks_prec, slesks_acc)

olesks_tkft_acc = olesks_tkft_acc/n_instances
olesks_prec = olesks_acc/olesks_n_rvd
olesks_acc = olesks_acc/n_instances
olesks_f1 = my_F1(olesks_prec, olesks_acc)

print("Base accuracy is: " + str(base_acc))
print("Simplified Lesk's algorithm, breaking 0 by first sense, accuracy/recall/precision/F1 is: " + str(slesks_tkft_acc))
print("Simplified Lesk's algorithm, ignoring 0's, accuracy/recall is: " + str(slesks_acc))
print("Simplified Lesk's algorithm, ignoring 0's, precision is: " + str(slesks_prec))
print("Simplified Lesk's algorithm, ignoring 0's, F1 is: " + str(slesks_f1))
print("Original Lesk's algorithm, breaking 0 by first sense, accuracy/recall/precision/F1 is: " + str(olesks_tkft_acc))
print("Original Lesk's algorithm, ignoring 0's, accuracy/recall is: " + str(olesks_acc))
print("Original Lesk's algorithm, ignoring 0's, precision is: " + str(olesks_prec))
print("Original Lesk's algorithm, ignoring 0's, F1 is: " + str(olesks_f1))

