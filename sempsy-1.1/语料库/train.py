
"""

Dependency:  https://pypi.org/project/gensim/

"""

# -*- coding: utf-8 -*-
import logging
from gensim.models import word2vec

def vector_train(corpus,size_num=300,window_num=10,min_count_num=5,sg_num=0):
    #Training log display
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)    
    
    sentences = word2vec.Text8Corpus(corpus)              
    model = word2vec.Word2Vec(sentences, vector_size=size_num, window=window_num, min_count=min_count_num,sg=sg_num)   #训练CBOW模型，维度300
    model.init_sims(replace=True)  
    model.save("word_vector.model")
    
    name = ['word_vector','win'+str(window_num),str(size_num)+'d']
    '_'.join(name)
    
    model.wv.save_word2vec_format('_'.join(name)+'.bin', binary=True)
    

    
    