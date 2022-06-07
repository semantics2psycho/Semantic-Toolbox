# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:39:42 2022

@author: Dell
"""
from cluster.tfidf import *
from semdistance.distance import *

filepath = 'E:/Semantic/Story/calculation/textcut/raw/story1txt'
textcut_outpath = 'E:/Semantic/Story/calculation/textcut/cut/story1'
weight_vector_outpath = 'E:/Semantic/Story/calculation/weight_vector'
tfidf_outpath = 'E:/Semantic/Story/calculation/tfidf'

[corpus,word_tfidf] = word_tfidf_gensim(filepath,'document')

outfile = []
# file = os.listdir(filepath)[0]
    
for file in os.listdir(filepath):
    outfile.append(os.path.join(tfidf_outpath,file))
    
for i in range(len(corpus)):
    text_out = []
    content = ' '.join(corpus[i])
    text_out.append(content)
    savefile(text_out,outfile[i],'.txt')
    
for i in range(len(weight_vector)):
    savefile(weight_vector[i],outfile[i],'.txt')

for i in range(len(word_tfidf)):
    savefile(word_tfidf[i],outfile[i],'.txt')
