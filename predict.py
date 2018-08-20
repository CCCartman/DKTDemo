# -*- coding: utf-8 -*-
# @Author  : Rui Wenhao
# @Email   : rwhcartman@163.com

import time
import sys
from DKTModel import *
from dataProcess import *
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import numpy as np
import os

def predict():
    dataset_path = 'D:\\mygit\\myDKT\\' + 'data\\skill_id_test.csv'

    test_seqs, num_skills = read_file(dataset_path)
    _ , test_seqs = split_dataset(test_seqs, sample_rate=1.0, random_seed=1)
    #print(test_seqs)
    batch_size = 50
    #print(batch_size,num_skills)
    test_generator = DataGenerator(test_seqs, batch_size=batch_size, num_skills=num_skills)
    test_generator.reset()

    config = {"hidden_neurons": [200],
              "batch_size": batch_size,
              "keep_prob": 0.7,
              "num_skills": num_skills,
              "input_size": num_skills * 2}

    model = DKTModel(config)

    gpuConfig = tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.per_process_gpu_memory_fraction = 0.6  # 限制一个进程使用 60% 的显存

    sess = tf.Session(config=gpuConfig)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    model_file = tf.train.latest_checkpoint('D:\\mygit\\myDKT\\' + 'ckptSkill\\')
    saver.restore(sess, model_file)

    preds, binary_preds, targets = [], [], []
    while not test_generator.end:
        input_x, target_id, target_correctness, seqs_len, max_len = test_generator.next_batch()
        binary_pred, pred, _ = model.step(sess, input_x, target_id,
                                    target_correctness,seqs_len, is_train=False)
        for seq_idx, seq_len in enumerate(seqs_len):
            preds.append(pred[seq_idx, :seq_len])
            binary_preds.append(binary_pred[seq_idx, :seq_len])
            targets.append(target_correctness[seq_idx, :seq_len])

    preds = np.concatenate(preds)
    binary_preds = np.concatenate(binary_preds)
    targets = np.concatenate(targets)
    auc_value = roc_auc_score(targets, preds)
    accuracy = accuracy_score(targets, binary_preds)
    precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)
    print("\n auc={0}, accuracy={1}, precision={2}, recall={3}".format(auc_value, accuracy,
               precision,recall))

if __name__ == '__main__':
    predict()

