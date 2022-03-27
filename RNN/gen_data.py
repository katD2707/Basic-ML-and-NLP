from collections import defaultdict
from email.policy import default
from genericpath import isfile
from os import listdir
import regex as re

def gen_data_and_vocab(path):
    def collect_data_from(parent_path, newsgroup_list, word_count=None):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            dir_path = parent_path + '/' + newsgroup + '/'
            files = [(filename, dir_path+filename) for filename in listdir(dir_path) \
                                                                    if isfile(dir_path + filename)]
            files.sort()
            label = group_id
            print('Processing: {} - {}'.format(group_id, newsgroup))
            for filename, file_path in files:
                with open(file_path) as f:
                    text = f.read().lower()
                    words = re.split('\W+', text)
                    if word_count is not None:
                        for word in words:
                            word_count[word] += 1
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' + filename + '<fff>' + content)
        return data
    word_count = defaultdict(int)
    
    parts = [path + file_name + '/' for file_name in listdir(path)  \
                                if not isfile(path + file_name)]
    
    train_path, test_path = (parts[0], parts[1]) if 'train' in parts[0]  \
                                                            else (parts[1], parts[0])

    newsgroup_list = [newsgroup for newsgroup in listdir(train_path)]

    newsgroup_list.sort()

    train_data = collect_data_from(parent_path=train_path, newsgroup_list=newsgroup_list, word_count=word_count)
    
    vocab = [word for word, freq in zip(word_count.keys(), word_count.values()) if freq > 10]
    vocab.sort()
    with open(path + '../../' + 'datasets/W2V/vocab-raw.txt', 'w') as f:
        f.write('\n'.join(vocab))

    test_data = collect_data_from(parent_path=test_path, newsgroup_list=newsgroup_list)
    
    with open(path + '../../' + 'datasets/W2V/20news-train-raw.txt', 'w') as f:
        f.write('\n'.join(train_data))
    with open(path + '../../' + 'datasets/W2V/20news-test-raw.txt', 'w') as f:
        f.write('\n'.join(test_data))

def encode_data(data_path, vocab_path):
    MAX_DOC_LENGTH = 500

    with open(vocab_path) as f:
        vocab = dict([(word, word_ID + 2) 
                        for word_ID, word in enumerate(f.read().splitlines())])
    with open(data_path) as f:
        documents = [(line.split('<fff>')[0], line.split('<fff>')[1], line.split('<fff>')[2]) for line in f.read().splitlines()]

    encoded_data = []
    for document in documents:
        label, doc_id, text = document
        words = text.split()[:MAX_DOC_LENGTH]
        sentence_length = len(words)
        encoded_text = []
        for word in words:
            if word in vocab:
                encoded_text.append(str(vocab[word]))
            else:
                encoded_text.append(str(1))
        
        if len(words) < MAX_DOC_LENGTH:
            num_paddings = MAX_DOC_LENGTH - len(words)
            for _ in range(num_paddings):
                encoded_text.append(str(0))

        encoded_data.append(str(label) + '<fff>' + str(doc_id) + '<fff>' +
                            str(sentence_length) + '<fff>' + ' '.join(encoded_text))

    dir_name = '/'.join(data_path.split('/')[:-1])
    file_name = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt'
    with open(dir_name + '/' + file_name, 'w') as f:
        f.write('\n'.join(encoded_data))

gen_data_and_vocab('../20news-bydate/')
encode_data(data_path='../datasets/W2V/20news-train-raw.txt', 
            vocab_path='../datasets/W2V/vocab-raw.txt')
encode_data(data_path='../datasets/W2V/20news-test-raw.txt', 
            vocab_path='../datasets/W2V/vocab-raw.txt')
