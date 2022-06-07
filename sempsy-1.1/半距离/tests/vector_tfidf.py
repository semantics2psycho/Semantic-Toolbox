# -*- coding: utf-8 -*-
"""
Created on Sun May 22 19:42:25 2022

@author: Dell
"""

from cluster.tfidf import *
from semdistance.distance import *

filepath = 'E:/Semantic/Story/calculation/textcut/raw/story1txt'
outpath = 'E:/Semantic/Story/calculation/textcut/cut/story1'

special_words = open(os.path.join('E:/Semantic/Story/calculation/textcut/specialwords.txt'), encoding ='utf-8').read().split('\n')
jieba.load_userdict(special_words)

[word_tf,word_idf,word_tfidf] = word_tfidf_sklearn(filepath,'document')
[corpus,word_tfidf] = word_tfidf_gensim(filepath,'document')

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

