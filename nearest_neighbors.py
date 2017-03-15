"""
Classifier for predicting morphological tags
"""
from dataset import Dictionary, load_corpus
from sklearn import preprocessing

from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from scipy.spatial.distance import cosine
from scipy.spatial.distance import hamming


import collections
import cPickle
import numpy as np
import theano
import theano.tensor as T
import math
from future_builtins import zip
import time
import logging

logger = logging.getLogger(__name__)

theano.config.blas.ldflags = "-L/export/apps/lib64/atlas/ " + theano.config.blas.ldflags
theano.config.exception_verbosity = "high"

    
def main():

    logger.info("Creating vocabulary dictionary...")
    vocab = Dictionary.from_corpus(train_data, unk='<unk>')
    logger.info("Creating tag dictionary...")
    vocab_tags = Dictionary.from_corpus_tags(train_data, unk='<unk>')
    vocab.add_word('<s>')
    vocab.add_word('</s>')
    V = vocab.size()
    
    vocab_tags.add_word('<s>')
    vocab_tags.add_word('</s>')
    V_tag = vocab_tags.size()
    
    feature_matrix = np.zeros((vocab_tags.size(),vocab_tags.num_sub_tags))
    feature_matrix[(0,0)] = 1 # unk encoding
    
    for tag,tag_id in vocab_tags:
        if tag == "<s>":
            feature_matrix[(tag_id,1)] = 1
        elif tag == "</s>":
            feature_matrix[(tag_id,2)] = 1
        else:
            for sub_tag in vocab_tags.map_tag_to_sub_tags[tag]:
                val = vocab_tags.map_sub_to_ids[sub_tag]
                feature_matrix[(tag_id,val)] = 1
    

    Q =  cPickle.load(open(sys.argv[4],'rb'))
  
    print "START COMPARING"

    word = sys.argv[5]
    word_id = vocab.lookup_id(word)
   
    words = []
    for j,q in enumerate(Q):
        words.append((j,vocab.lookup_word(j),cosine(Q[word_id],q)))
        words.sort(key=lambda x:x[2])
    print words[:20]

if __name__ == '__main__':
    import sys
    from docopt import docopt

    def read_in_data(file_in):
        # ryan's reading in of the morphologically rich data
        
        counter = 0
        data = []
        with open(file_in, 'rb') as fin:
            cur_sentence = []
            for line in fin.readlines():
                line = line.rstrip("\n")
                
                if line == "":
                    # line break
                    data.append(cur_sentence)
                    cur_sentence = []
                else:
                    word,tag = line.split("\t")
                    cur_sentence.append((word,tag))
            
            data.append(cur_sentence)
            
        return data
    
    logger.info("Reading in training data...")
    train_data = read_in_data(sys.argv[1])
    
    v_to_tags = collections.defaultdict(list)
    v_to_tag = {}
   
    tag_set = {}
    num_tags = 0
   
    word_set = {}
    num_words = 0

    for sentence in train_data:
        for word,tag in sentence:
            v_to_tags[word].append(tag)

            if tag not in tag_set:
                tag_set[tag] = num_tags
                num_tags += 1

            if word not in word_set:
                word_set[word] = num_words
                num_words += 1

    for word,tags in v_to_tags.items():
        tag = max(set(tags),key=tags.count)
        v_to_tag[word] = tag

    logger.info("Reading in dev data...")
    dev_data = read_in_data(sys.argv[2])
    logger.info("Reading in test data...")
    test_data = read_in_data(sys.argv[3])

    # main method
    main()
