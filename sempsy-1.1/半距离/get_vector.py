# -*- coding: utf-8 -*-
"""
Created on Wed May 25 17:16:19 2022

@author: Dell
"""
import gensim
import numpy as np
from scipy.spatial.distance import pdist
from textprocess.textcut import *
from semdistance.model_path import *
from cluster.tfidf import *

# get vector model
def get_model(model_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_path), binary=True)
    return model

# word vector model
model_test = get_model(model_test_path)
model_trained = get_model(model_trained_path)

# get word vector
def get_word_vector(word,have_model):
    if have_model == 0:
        model = model_test
    else:
        model = model_trained
    try:
        return model[word]
    except:
        try:
            word_list = textcut(word)
            vector_list = [model[w] for w in word_list]
            return np.mean(vector_list,0)

        except:
            print(word)
            print('该词语在词典中未检测到！')
            return np.zeros(300)[: int(300)]
            # return 

# unit vector calculation
def _unitvec(v): return v/ np.linalg.norm(v)

# get word sequence vector
def get_word_sequence_vector(word_sequence,model=0):
    vector_list = []
    for word in word_sequence:
        v = get_word_vector(word, model)
        if not(np.all(v==0)):
            vector_list.append(v)
    return vector_list


# get relation vector by subtracting word vectors from word pairs  
def get_relation_vector(pair,model=0):
        vec1,vec2 = [get_word_vector(p,model) for p in pair]
        relation_vector = np.subtract(vec1,vec2)
        return relation_vector


# get sentence vector by averaging word vector in the sentence
# parameter:
# s: input data
# model: 0,默认路径；1，model_path中放训练好的词向量路径
# preprocess: 0，为预处理的文本；1预处理后的文本-list格式
def get_sentence_vector(s, model=0, preprocess=0):
    if preprocess == 1:
        vectorized_sentence = [get_word_vector(w,model) for w in s]
    else:
        vectorized_sentence = [get_word_vector(w,model) for w in textcut(s)]
    sentence_vector = np.mean(vectorized_sentence, 0)
    return sentence_vector


def get_tfidf_weighted_vector(filepath,tfidf='gensim',split='document',model=0):

    # filepath = 'E:/Semantic/Story/calculation/textcut/raw/story1txt'
    # outpath = 'E:/Semantic/Story/calculation/textcut/cut/story1'
    
    # special_words = open(os.path.join('E:/Semantic/Story/calculation/textcut/specialwords.txt'), encoding ='utf-8').read().split('\n')
    # jieba.load_userdict(special_words)
    
    if tfidf == 'gensim':
        if split == 'document':
            [corpus,word_tfidf] = word_tfidf_gensim(filepath,'document')
        elif split == 'wordLength':
            [corpus,word_tfidf] = word_tfidf_gensim(filepath,'wordlength')
            
    elif tfidf == 'sklearn':
        if split == 'document':
            [corpus,word_tfidf] = word_tfidf_sklearn(filepath,'document')
        elif split == 'wordLength':
            [corpus,word_tfidf] = word_tfidf_sklearn(filepath,'wordLength')
        

    word_vector_list = []
    # words_list = corpus[0]
    for words_list in corpus:
        vector_list = []
        word_vector_sub = []
        vector_list = get_word_sequence_vector(words_list,1)
        word_vector_sub = {words_list[i]:vector_list[i] for i in range(len(vector_list))}
        word_vector_list.append(word_vector_sub)
    
    
    # i = 0
    # j = 0
    vector_tfidf_list = []
    for i in range(len(corpus)):
        vector_tfidf = []
        corpus_sub = corpus[i]
        word_vector_sub = word_vector_list[i]
        tfidf_sub = word_tfidf[i]
        
        w = corpus_sub[0]
        for w in corpus_sub:
            vector_tfidf.append(np.dot(word_vector_sub[w],tfidf_sub[w]))

        vector_tfidf_list.append(vector_tfidf)
    return vector_tfidf_list