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
import read_author


class AuthorModel(object):
    def __init__(self, config_param, vocab_size):
        tf.reset_default_graph()
        self.config = config_param
        self.vocab_size = vocab_size

        self.build_author_model()

        #session here, then call run model.
        self.sess = tf.Session()

        self.run_author_model()
 

    def build_author_model(self):
        self.writer = tf.summary.FileWriter("/tmp/jane/3")
        self.X_ph = tf.placeholder(tf.int32, [None,None], "InputsX")
        self.Y_ph = tf.placeholder(tf.int32, [None, None], "InputTargetsY")
        self.is_training = tf.placeholder(tf.bool)
        
        #embedding for each character in vocabulary
        embedding = tf.get_variable("embedding", [self.vocab_size, self.config.embed_size])
        embeddingLookedUp = tf.nn.embedding_lookup(embedding, self.X_ph)
                
        #Define Tensor RNN
        singleRNNCell = tf.contrib.rnn.GRUCell(self.config.rnn_size)
        multilayerRNN =  tf.nn.rnn_cell.MultiRNNCell([singleRNNCell] * self.config.num_layers)

        
        hidden_layer_output, last_state = tf.nn.dynamic_rnn(multilayerRNN, embeddingLookedUp,dtype=tf.float32)#, initial_state=self.initial_state)
 
        hidden_layer_output = tf.reshape(hidden_layer_output,[-1,self.config.rnn_size])


        logits = tf.nn.xw_plus_b(hidden_layer_output, tf.get_variable("softmax_w", [self.config.rnn_size, self.vocab_size]), tf.get_variable("softmax_b", [self.vocab_size]))

        #allow for no steps in back prop
        first_dim = tf.cond(self.is_training,lambda: self.config.num_steps,lambda: 1)

        #reshape for seq2seq loss
        logits = tf.reshape(logits,[first_dim,-1,self.vocab_size])
        
        #save for prediction
        self.predictionSoftmax = tf.nn.softmax(logits)

        #Define the loss
        self.cost = tf.contrib.seq2seq.sequence_loss(logits,self.Y_ph,tf.ones([self.config.batch_size, self.config.num_steps]))
        #self.cost = tf.div(tf.reduce_sum(self.loss), self.config.batch_size)
        self.final_state = last_state


        trainingVars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainingVars),self.config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train = optimizer.apply_gradients(zip(grads, trainingVars))
        #t1 = time.time()
        


        
    def run_author_model(self):
        self.writer.add_graph(self.sess.graph)
        self.outfile = open(self.config.outfile_name+'.txt','w',encoding='utf-8')
        self.pltfile = open(self.config.outfile_name+'_plts.txt','w',encoding='utf-8')
        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        for epochCount in range(self.config.num_epochs):
            accumulated_costs = 0.0
            accumulated_seq_count = 0
            books_reader = read_author.BooksReader(self.config.file_path)
            current_model_state=np.array([np.array([0]*self.config.rnn_size)]*self.config.batch_size)

            lowest_perplexity = 2000
            sum_costs=0
            for sequence_counter, (x, y) in enumerate(books_reader.generateXYPairs(books_reader.get_training_data(), self.config.batch_size, self.config.num_steps)):
                inX = np.transpose(np.array(x),[1,0])
                labY = np.transpose(np.array(y),[1,0])
                feed_dict = {self.X_ph: inX, 
                             self.Y_ph: labY,
                             self.is_training: True
                            } 
                cost, current_model_state, _ = self.sess.run([self.cost, self.final_state, self.train], feed_dict)
                accumulated_costs += cost
                sum_costs += cost
                
                accumulated_seq_count += self.config.num_steps
                perplexity =  np.exp(accumulated_costs / accumulated_seq_count)
                
                #print('run time ',t2-t1)
                if  sequence_counter != 0 and sequence_counter % 50 == 0:              
                    #if sequence_counter % 50 == 0:
                    self.outfile.write("Epoch %d, Perplexity: %.3f, Loss: %.3f" % (epochCount, perplexity,cost))
                    if perplexity < lowest_perplexity:
                        lowest_perplexity = perplexity
                        
                        saver.save(self.sess, os.path.join(os.getcwd(), self.config.outfile_name+'.ckpt')) 

                        t3 = time.time()
                        #print('here')
                        self.get_prediction(books_reader,  500, ['E','l','i','z','a','b','e','t','h',' '])
                        t4 = time.time()
                        #print('save time, pred time = ',t3-t2,t4-t3)
            avgloss = sum_costs/sequence_counter
            t5 = time.time()
            plt_line = str(epochCount)+','+str(avgloss)+','+str(perplexity)+','+'\n'
            self.pltfile.write(plt_line)
            #average_losses.append(sum_costs/countt)
            if epochCount % 10 == 0 :
                self.outfile.write("Epoch %d, Perplexity: %.3f, Loss: %.3f" % (epochCount, perplexity,cost))
                self.get_prediction(books_reader,  500, ['E','l','i','z','a','b','e','t','h',' '])
            self.pltfile.flush()
            self.outfile.flush()
            os.fsync(self.pltfile.fileno())
            os.fsync(self.outfile.fileno())
            t6 = time.time()
            #print('file time ',t6-t5 ) 
        self.outfile.close()
        self.pltfile.close()                
        
    def get_prediction(self, books_Reader, total_tokens, output_tokens = ['']):
        #state = self.multilayerRNN.zero_state(1, tf.float32).eval()#
        state=np.array([np.array([0]*self.config.rnn_size)]*1)

        for token_count in range(total_tokens):
            next_token = output_tokens[token_count]
            input1 = np.full((1, 1), books_Reader.token_to_id[next_token], dtype=np.int32)
            feed = {self.X_ph: input1,self.is_training: False}#, self.initial_state:state}
            [predictionSoftmax] =  self.sess.run([self.predictionSoftmax], feed)
            #print(predictionSoftmax)
            if (len(output_tokens) -1) <= token_count:
                accumulated_sum = np.cumsum(predictionSoftmax[0]/self.config.temp)
                currentTokenId = (int(np.searchsorted(accumulated_sum, np.random.rand(1))))
                #print(currentTokenId)
                next_token = books_Reader.unique_tokens[currentTokenId]
                output_tokens.append(next_token)

        output_sentence = ""
        for token in output_tokens:
            output_sentence+=token
        self.outfile.write('---- Prediction: \n %s \n----' % (output_sentence))

        return
        
        
        
    def close_sess(self):
        self.sess.close()
        
        
        

        
