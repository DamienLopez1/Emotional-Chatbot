from __future__ import unicode_literals

import pandas as pd
from barbar import Bar
import torch
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
from itertools import chain
import json
import torch.nn as nn
import os
import re
import unicodedata
import torch.optim as optim
import torch.nn.functional as F
import random
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader, random_split,RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt



from transformers import cached_path
import tarfile
import tempfile
import logging

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)



from transformers import BertTokenizer, BertForSequenceClassification,AdamW
from transformers import get_linear_schedule_with_warmup


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels = 5, output_attentions = False, output_hidden_states = False,hidden_dropout_prob = 0.4)



model.to(device)

#createdataset



datafile = 'dialogues_text2.txt'
emodatafile = 'dialogues_emotion2.txt'

bos, eos, speaker1, speaker2, CLS  = "<BOS>", "<EOS>", "<speaker1>", "<speaker2>", "<CLS>"

def normalizeString(s): #normailizes string
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s



def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def create_dialogues2(datafile,emodatafile): #takes text and puts it into a list 
    emopairs = []
    lines = open(datafile, encoding='utf-8').\
            read().strip().split('\n')
            
    emolines = open(emodatafile, encoding='utf-8').\
            read().strip().split('\n')
        # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('__eou__')] for l in lines]
    for i in range(len(pairs)):
        if pairs[i][-1] == '':
            pairs[i] = pairs[i][:-1] #removes empty character and keeps rest
        emopairs.append(emolines[i].split(' '))
        if emopairs[-1][-1] == '': 
            emopairs[-1] = emopairs[-1][:-1]
    
    return pairs,emopairs



def createdata(pairs,emotionpairs):
    output = []
    emotions = []
    for line in range(len(pairs)):        
        for j in range(len(pairs[line])):
            if len(tokenizer.encode(pairs[line][j])) <= 78:
                output.append(pairs[line][j])
                emotions.append(emotionpairs[line][j])
            

    return output,emotions


def balancedata(pairs,emotionpairs, maxbal):
    distribution = [0,0,0,0,0,0,0]

    for labely in range(7):
        for count in emotionpairs:
            #print(count)
            #print(labely)
            if int(count) == labely:
                #print('yes')
                distribution[labely] += 1
    
    print(distribution)


    newpairs = []
    newemotions = []
    distribution = [0,0,0,0,0,0,0]

    for count in range(len(emotionpairs)):
        if distribution[int(emotionpairs[count])] < maxbal and int(emotionpairs[count]) != 2 and int(emotionpairs[count]) != 3:
                distribution[int(emotionpairs[count])] += 1
                newpairs.append(pairs[count])
                newemotions.append(emotionpairs[count])
    
    print(len(newemotions))


    distribution = [0,0,0,0,0,0,0]
    
    for labely in range(7):
        for count in newemotions:
            #print(count)
            #print(labely)
            if int(count) == labely:
                #print('yes')
                distribution[labely] += 1
    
    print(distribution)
    
    for i in range(len(newemotions)):
        if int(newemotions[i]) > 1:
            newemotions[i] = str(int(newemotions[i]) - 2)
    
    pairs = newpairs
    emotionpairs = newemotions
     
    
    distribution = [0,0,0,0,0]
    
    for labely in range(5):
        for count in newemotions:
            #print(count)
            #print(labely)
            if int(count) == labely:
                #print('yes')
                distribution[labely] += 1  
    
    print(distribution)
    return newpairs,newemotions
    
    
#make emotion and utterance pairs. Max 2000 of each emotion, remove afraid and disgusted emotions
pairs,emotionpairs = create_dialogues2(datafile,emodatafile)
pairs,emotionpairs = createdata(pairs,emotionpairs)
pairs,emotionpairs = balancedata(pairs,emotionpairs,2000) 

#tokenize and create train,validation and test set

input_ids = []
attention_masks = []
labels = []
 
for i in range(len(pairs)):
            
    encoded_dict = tokenizer.encode_plus(
                pairs[i],                      # Sentence to encode.
                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                max_length = 78,           # Pad & truncate all sentences.
                pad_to_max_length = True,
                return_attention_mask = True,   # Construct attn. masks.
                return_tensors = 'pt',     # Return pytorch tensors.
           )
    
    input_ids.append(encoded_dict['input_ids'])
    
    # attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])
    labels.append(int(emotionpairs[i]))
    
# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

big_dataset = TensorDataset(input_ids, attention_masks, labels)

train, valid, test = torch.utils.data.random_split(big_dataset, [8000,1000,830])


train_dataloader = DataLoader(
            train,  # The training samples.
            sampler = RandomSampler(train), # Select batches randomly
            batch_size = 32 # Trains with this batch size.
        )


Val_dataloader = DataLoader(
            valid,  # The validation samples.
            sampler = SequentialSampler(valid), # Select batches sequentially
            batch_size = 32 # Trains with this batch size.
        )



test_dataloader = DataLoader(
            test,  # The test samples.
            sampler = SequentialSampler(test), # Select batches sequentially
            batch_size = 32 # Trains with this batch size.
        )





#optimizer

epochs = 10



optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
                  weight_decay = 0.4
                )


total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


#train

   
def train_batch():
    min_loss = 10 
    for i in range(epochs):
        print('Epoch number', i)
        model.train()
        
        total_loss = 0
        total = 0
        numbersamples = 0
        preds = []
        labs = []
        for id,X in enumerate(Bar(train_dataloader)):
            optimizer.zero_grad()
            
            input_ids,attention_mask, labels = X
            
        
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            
            outputs = model(input_ids,token_type_ids=None,attention_mask = attention_mask, labels=labels)
            loss, logits = outputs[:2]
            
            loss = loss 
            
            total_loss += loss.item()
            loss.backward()

            _ = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            
            optimizer.step()
            #scheduler.step()
            
            for i in range(len(labels)):
                values, indices = torch.max(outputs[1][i],0)
                preds.append(indices.item())
                labs.append(labels[i].item())
                if indices.item() == labels[i].item():
                    total += 1
                    
        
            numbersamples += len(labels)
        print('Train Accuracy', total/numbersamples)
        print('Train loss' , total_loss/len(train_dataloader))
        
        
        
        model.eval()
        total_eval_loss = 0
        total = 0
        numbersamples = 0
        preds = []
        labs = []
        for batch in Val_dataloader:
            
            input_ids,attention_mask, labels = batch
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():   
                outputs = model(input_ids,token_type_ids=None,attention_mask = attention_mask, labels=labels)
                loss, logits = outputs[:2]
                
            for i in range(len(labels)):
                values, indices = torch.max(outputs[1][i],0)
                preds.append(indices.item())
                labs.append(labels[i].item())
                if indices.item() == labels[i].item():
                    total += 1
                    
                
            total_eval_loss += loss.item()
            numbersamples += len(labels)
        
        print('Val Accuracy', total/numbersamples)
        print('Val loss', total_eval_loss/len(Val_dataloader))
        if total_eval_loss/len(Val_dataloader) < min_loss:
            min_loss = total_eval_loss/len(Val_dataloader)
            output_dir = './Sentiment model - Saved/'
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        
train_batch()

#evaluate


tokenizer = BertTokenizer.from_pretrained('./Sentiment model - Saved/')
model = BertForSequenceClassification.from_pretrained('./Sentiment model - Saved/',num_labels = 5, output_attentions = False, output_hidden_states = False,hidden_dropout_prob = 0.4)
model.to(device)

#Uncomment below code to type in sentences and see resulting emotion
'''
model.eval()
personalities = ["I have no emotion", "I am Angry", "I am happy", "I am sad", "I am surprised"]
while True:
        raw_text = input(">>> ")
        if raw_text == 'q': 
                break
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
            if raw_text == 'q': 
                break


        encoded_dict = tokenizer.encode_plus(
                        raw_text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 78,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
        
        input_ids = [encoded_dict['input_ids']]
        attention_masks = [encoded_dict['attention_mask']]
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        
        input_ids = input_ids.to(device)
        attention_mask = attention_masks.to(device)
        
        outputs = model(input_ids,token_type_ids=None,attention_mask = attention_mask)
        values, indices = torch.max(outputs[0][0],0)
        
        print('emotion is ' , indices)
        
        print(personalities[indices.item()])
'''

# Evaluate classifier on test set. Accuracy metric and Confusion matrix.
total = 0    
numbersamples = 0
preds = []
labs = []
model.eval()

for batch in test_dataloader:
    
    input_ids,attention_mask, labels = batch
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    
    with torch.no_grad(): 
        outputs = model(input_ids,token_type_ids=None,attention_mask = attention_mask)
    
    
    for i in range(len(labels)):
        values, indices = torch.max(outputs[0][i],0)
        preds.append(indices.item())
        labs.append(labels[i].item())
        if indices.item() == labels[i].item():
            total += 1

    
    numbersamples += len(labels)


cm = confusion_matrix(labs, preds)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print(cmn)

print(total/numbersamples)  
print(confusion_matrix(labs, preds))
sn.heatmap(cmn,cmap = 'Reds', annot=True)  
plt.ylabel('Actual')
plt.xlabel('Predicted')
#plt.savefig('./SentimentCM.png')
