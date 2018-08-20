# -*- coding: utf-8 -*-
"""

@author: Rui Wenhao
@Mail:rwhcartman@163.com

"""

import time
import sys
from DKTModel import *
from dataProcess import *
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import numpy as np
import os

def run():
    dataset_path = 'D:\\mygit\\myDKT\\' + 'data\\skill_id_train.csv'
    seqs_by_student, num_skills = read_file(dataset_path)
    train_seqs,test_seqs = split_dataset(seqs_by_student)
    
    batch_size = 10
    train_generator = DataGenerator(train_seqs, batch_size=batch_size, num_skills=num_skills)
    test_generator = DataGenerator(test_seqs, batch_size=batch_size, num_skills=num_skills)
    
    config = {"hidden_neurons": [200],
              "batch_size": batch_size,
              "keep_prob": 0.7,
              "num_skills": num_skills,
              "input_size": num_skills * 2}

    model = DKTModel(config)
    #tf.reset_default_graph()
    gpuConfig = tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.per_process_gpu_memory_fraction = 0.6 # 限制一个进程使用 60% 的显存

    sess = tf.Session(config=gpuConfig)
    sess.run(tf.global_variables_initializer())
    print(tf.trainable_variables())
    lr = 0.5
    lr_decay = 0.92
    saver = tf.train.Saver()

    max_auc_value = 0
    f = open('D:\\mygit\\myDKT\\' + 'ckptSkill\\dkt_auc.txt','w')
    for epoch in range(10):
        model.assign_lr(sess,lr * lr_decay ** epoch)
        #overall_loss = 0
        train_generator.shuffle()
        start = time.time()
        while not train_generator.end:
            input_x, target_id, target_correctness, seqs_len, max_len = train_generator.next_batch()
            loss_ = model.step(sess, input_x, target_id, target_correctness,\
                                           seqs_len, is_train=True)
            print("\r epoch:{0}, idx:{1},loss_:{2}, time spent:{3}s".format(str(epoch+1),\
                    train_generator.pos, loss_,time.time() - start))
            sys.stdout.flush()
        preds, binary_preds, targets =[],[],[]
        while not test_generator.end:
            input_x, target_id, target_correctness, seqs_len, max_len = test_generator.next_batch()
            binary_pred, pred, _ = model.step(sess, input_x, target_id, target_correctness,\
                                              seqs_len, is_train=False)
            for seq_idx,seq_len in enumerate(seqs_len):
                preds.append(pred[seq_idx,:seq_len])
                binary_preds.append(binary_pred[seq_idx,:seq_len])
                targets.append(target_correctness[seq_idx,:seq_len])
        preds = np.concatenate(preds)
        binary_preds = np.concatenate(binary_preds)
        targets = np.concatenate(targets)
        auc_value = roc_auc_score(targets, preds)
        accuracy = accuracy_score(targets, binary_preds)
        precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)
        print("\n epoch = {0},auc={1}, accuracy={2}, precision={3}, recall={4}".format(str(epoch+1),\
              auc_value,accuracy, precision, recall))
        f.write(str(epoch+1)+'\t'+str(auc_value)+'\n')
        if auc_value > max_auc_value:
            max_auc_value = auc_value
            saver.save(sess, 'D:\\mygit\\myDKT\\' + 'ckptSkill\\dkt.ckpt',\
                           global_step=epoch+1)
        train_generator.reset()
        test_generator.reset()
    f.close()
    
if __name__ == '__main__':
    run()