import tensorflow as tf
import collections
import os
import os.path
import sys
import re
import time
import numpy as np
from tensorflow.contrib import seq2seq
import AuthorConfig








class BooksReader(object):
    def __init__(self, data_path):
        
        #call a function here to read all authors books into single string
        openfile = open(data_path,encoding="UTF-8")
        uncleaned_data = openfile.read()
        openfile.close()

        #remove excess whitespaces etc with clean_data()
        self.raw_data = self.clean_data(uncleaned_data)
        
        #Build Vocabulary and Dictionaries
        counter = collections.Counter(self.raw_data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.unique_tokens, _ = list(zip(*count_pairs))
        self.token_to_id = dict(zip(self.unique_tokens, range(len(self.unique_tokens))))
        self.vocabularySize = len(self.unique_tokens)
        
        #Convert Raw tokens to digits
        self.data_as_ids = []
        for place, token in enumerate(self.raw_data):
            self.data_as_ids.append(self.token_to_id[token])
            
        self.data_size = len(self.data_as_ids)
        self.train_data_max_index = int(round(self.data_size*.9))
        self.valid_data_min_index = 0#self.train_data_max_index+1
        self.valid_data_max_index =0# self.valid_data_min_index + int(round(self.data_size * self.val_percent))

    
    def clean_data(self, raw_data):
        data_remove_chapter = re.sub(r'Chapter', ' ', raw_data)
        data_remove_newline = re.sub(r'\n\s*\n', '\n',data_remove_chapter)
        data_remove_cr = re.sub(r'(?<=[a-z])\r?\n',' ',data_remove_newline)
        data_remove_cr1 = re.sub(r'(?<=,)\r?\n',' ',data_remove_cr)
        data_remove_cr2 = re.sub(r'(?<=;)\r?\n',' ',data_remove_cr1)
        data_remove_num = re.sub('\d+', '', data_remove_cr2)
        data_remove_under = re.sub('_', '', data_remove_num)#data_remove_num.replace('_', '')
        return data_remove_under
        
        
    def print_data_info(self):
        print('----------------------------------------')
        print('Data total tokens: %d tokens' % (len(self.raw_data)))
        print('Data vocabulary size: %d tokens' % (len(self.unique_tokens)))
        print('Training Data total tokens: %d tokens' % (len(self.get_training_data())))
        print('----------------------------------------')

    def get_training_data(self):
        return self.data_as_ids[0:self.train_data_max_index]
        
    def generateXYPairs(self, raw_data, batch_size, num_steps):
        #iterator through the batch 
        raw_data = np.array(raw_data, dtype=np.int32)

        num_batches = len(raw_data) // batch_size
        data = np.zeros([batch_size, num_batches], dtype=np.int32)
        for i in range(batch_size):
            data[i] = raw_data[num_batches * i:num_batches * (i + 1)]

        for i in range((num_batches - 1) // num_steps):
            x = data[:,i*num_steps:(i+1)*num_steps]
            y = data[:,i*num_steps+1:(i+1)*num_steps+1]
            yield (x, y)



    
