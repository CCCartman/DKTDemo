import numpy as np
import csv
import random


def read_file(dataset_path):
    seqs_by_student = {} ## {'1':[[82,0],[82,0],[82,0]]}
    num_skills = 0
    if dataset_path.endswith('.txt'):
        with open(dataset_path, 'r') as f:
            for line in f:
                fields = line.strip().split()
                student, problem, is_correct = int(fields[0]), int(fields[1]), int(fields[2])
                num_skills = max(num_skills, problem)
                seqs_by_student[student] = seqs_by_student.get(student, []) + [[problem, is_correct]]
    elif dataset_path.endswith('.csv'):
        rows = []
        with open(dataset_path,'r') as f:
            reader = csv.reader(f, delimiter = ',')
            for row in reader:
                rows.append(row)
        student = 0
        for i in range(0,len(rows),3):
            seq_length = int(rows[i][0])
            if seq_length < 2:
                continue
            problem_seq = rows[i + 1]
            correct_seq = rows[i + 2]
            invalid_loc = [i for i,pid in enumerate(problem_seq) if pid == '']
            
            for _loc in invalid_loc:
                del problem_seq[_loc]
                del correct_seq[_loc]
            problem_seq = list(map(int,problem_seq))
            correct_seq = list(map(int,correct_seq))
            sub_num_skills = max(problem_seq)
            num_skills = max(num_skills,sub_num_skills)
            problem_and_correct = list(zip(problem_seq,correct_seq))
            seqs_by_student[student] = seqs_by_student.get(student, []) + problem_and_correct
            student += 1
    return seqs_by_student, num_skills + 1


def split_dataset(seqs_by_student, sample_rate=0.2, random_seed=1):
    sorted_keys = sorted(seqs_by_student.keys()) ## 学生编号
    random.seed(random_seed)
    test_keys = set(random.sample(sorted_keys, 
                                  int(len(sorted_keys) * sample_rate))) ## 学生编号打乱顺序
    test_seqs = [seqs_by_student[k] for k in 
                 seqs_by_student if k in test_keys] ## test_keys中对应的做题记录
    train_seqs = [seqs_by_student[k] for k in  
                  seqs_by_student if k not in test_keys] ## train_keys中对应的做题记录
    return train_seqs, test_seqs


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    lengths = [len(s) for s in sequences] ## 每个用户回答问题数
    nb_samples = len(sequences) ## batch_size
    if maxlen is None:
        maxlen = np.max(lengths) ## 最多做了多少题

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:] ## 列的维度 [skill_id,TorF]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        else:
            x[idx] = np.pad(s, (0,maxlen - len(s)),'constant', constant_values=value)
    return x


def num_to_one_hot(num, dim):
    base = np.zeros(dim)
    if num >= 0: ## -1不设置
        base[num] += 1
    return base

def format_data(seqs, batch_size, num_skills):
    gap = batch_size - len(seqs) ## gap:对输入做题序列长度补0个数
    seqs_in = seqs + [[[0, 0]]] * gap  ## 补齐batch
    seq_len = np.array(list(map(lambda seq: len(seq), seqs_in))) - 1 ## Time Series，-1
    max_len = max(seq_len) ## 每个序列长度里最长的
    x = pad_sequences(np.array([[(j[0] + num_skills * j[1]) for j in i[:-1]] \
                                 for i in seqs_in]), maxlen=max_len, value=-1)
    ## 输入序列one-hot编码 左侧答错编码 右侧答对编码
    input_x = np.array([[num_to_one_hot(j, num_skills*2) for j in i] for i in x])

    ## 答题编号序列
    target_id = pad_sequences(np.array([[j[0] for j in i[1:]] for i in seqs_in]), 
                              maxlen=max_len, value=0)

    ## 答题结果序列
    target_correctness = pad_sequences(np.array([[j[1] for j in i[1:]] for \
                            i in seqs_in]), maxlen=max_len, value=0)
    return input_x, target_id, target_correctness, seq_len, max_len


class DataGenerator(object):
    def __init__(self, seqs, batch_size, num_skills):
        self.seqs = seqs
        self.batch_size = batch_size
        self.pos = 0
        self.end = False
        self.size = len(seqs)
        self.num_skills = num_skills

    def next_batch(self):
        batch_size = self.batch_size
        if self.pos + batch_size < self.size:
            batch_seqs = self.seqs[self.pos:self.pos + batch_size]
            self.pos += batch_size
        else:
            batch_seqs = self.seqs[self.pos:]
            self.pos = self.size - 1
        if self.pos >= self.size - 1:
            self.end = True
        input_x, target_id, target_correctness, seqs_len, max_len = \
                format_data(batch_seqs, batch_size, self.num_skills)
        return input_x, target_id, target_correctness, seqs_len, max_len

    def shuffle(self):
        self.pos = 0
        self.end = False
        np.random.shuffle(self.seqs)

    def reset(self):
        self.pos = 0
        self.end = False

if __name__ == '__main__':
    dataset_path = 'D:\\workspaceQiuzhao\\myDKT\\' + 'data\\skill_id_train.csv'
    seqs_by_student, num_skills = read_file(dataset_path)
    print(num_skills)