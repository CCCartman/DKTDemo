# -*- coding: utf-8 -*-
"""

@author: Rui Wenhao
@Mail:rwhcartman@163.com

"""

import tensorflow as tf

class DKTModel:
    def __init__(self,config):
        self.hidden_neurons = config["hidden_neurons"] ## 隐层神经元
        self.num_skills = config["num_skills"]
        self.input_size = config["input_size"] ## 输入数据 num_skills * 2
        self.batch_size = config["batch_size"] ## batch
        self.keep_prob_value = config["keep_prob"] ## drop_out
    
        # createPlaceHolder
        self.max_steps = tf.placeholder(tf.int32)  # max seq length of current batch
        self.input_data = tf.placeholder(tf.float32, [self.batch_size, None, self.input_size]) ## None 不知道做多少题
        self.sequence_len = tf.placeholder(tf.int32, [self.batch_size]) ## 每个人答题数 - 1
        self.keep_prob = tf.placeholder(tf.float32)  # dropout keep prob
        self.target_id = tf.placeholder(tf.int32, [self.batch_size, None]) ## 题目编号矩阵
        self.target_correctness = tf.placeholder(tf.float32, [self.batch_size, None]) ## 答题结果矩阵


        # createNet(self):
        hidden_layers = []
        for _,hidden_size in enumerate(self.hidden_neurons):
            lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size,state_is_tuple = True)
            hidden_layer = tf.contrib.rnn.DropoutWrapper(cell = lstm_layer,
                                                         output_keep_prob = self.keep_prob)
            hidden_layers.append(hidden_layer)
        hidden_cell = tf.contrib.rnn.MultiRNNCell(cells = hidden_layers,state_is_tuple = True)
        state_series,self.current_state = tf.nn.dynamic_rnn(cell = hidden_cell,
                                                       inputs = self.input_data,
                                                       sequence_length = self.sequence_len,
                                                       dtype = tf.float32)
        
        output_W = tf.get_variable(name = 'W',shape = [self.hidden_neurons[-1],self.num_skills])
        output_b = tf.get_variable(name = 'b',shape = [self.num_skills])
        
        state_series = tf.reshape(state_series,shape = [(self.batch_size * self.max_steps),\
                                                        self.hidden_neurons[-1]])
        
        self.logits = tf.add(tf.matmul(state_series,output_W),output_b)
        
        self.pred_all = tf.sigmoid(tf.reshape(self.logits,\
                                              [self.batch_size,self.max_steps,self.num_skills]))
        
        # computeLoss(self):

        flat_logits = tf.reshape(self.logits,[-1])
        flat_target_correctness = tf.reshape(self.target_correctness,[-1])
        flat_base_target_index = tf.range(self.batch_size * self.max_steps) * self.num_skills
        
        flat_target_id = tf.reshape(self.target_id,[-1])
        flat_target_id = flat_target_id + flat_base_target_index
        
        flat_target_logits = tf.gather(flat_logits,flat_target_id)
        
        self.pred = tf.sigmoid(tf.reshape(flat_target_logits,[self.batch_size,self.max_steps]))
        
        self.binary_pred = tf.cast(tf.greater_equal(self.pred,0.5),tf.int32)
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = flat_target_correctness,
                                                                          logits = flat_target_logits))
        
        self.lr = tf.Variable(0.0, trainable = False)
        
        #optimizer = tf.train.AdagradOptimizer(learning_rate = self.lr)
        #self.train_op = optimizer.minimize(self.loss)
        _vars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss,_vars), 4)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(self.grads, _vars))
            
    def step(self,sess,input_x,target_id,target_correctness,sequence_len,is_train):
        _, max_steps, _ = input_x.shape
        input_feed = {self.input_data:input_x,
                      self.target_id:target_id,
                      self.target_correctness:target_correctness,
                      self.max_steps:max_steps,
                      self.sequence_len: sequence_len}
        if is_train:
            input_feed[self.keep_prob] = self.keep_prob_value
            train_loss,_,_ = sess.run([self.loss,self.train_op,self.current_state],
                                      feed_dict = input_feed)
            return train_loss
        else:
            input_feed[self.keep_prob] = 1
            bin_pred,pred,pred_all = sess.run([self.binary_pred, self.pred, self.pred_all], \
                                              feed_dict = input_feed)
            return bin_pred,pred,pred_all
    
    def assign_lr(self, sess, lr_value):
        sess.run(tf.assign(self.lr, lr_value))