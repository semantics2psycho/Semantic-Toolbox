# -*- coding: utf-8 -*-
"""
Created on Thu May 19 20:05:38 2022

@author: PC
"""

'''LIWC'''

import os
# import liwc
import pandas as pd
import jieba
import glob

simple_stopwords = "D:\\学习\\自然主义\\word2vec python包\\Semantic-Toolbox-main\\Semantic-Toolbox-main\\sempsy-1.0\\textprocess\\tests\\simple_stopwords.txt"
with open(simple_stopwords, 'r',encoding="utf-8") as f:
    stopwords = [word.strip() for word in f.readlines()]
    
def textcut(text,stop=1,stopwords=stopwords):
    if stop is None:
        stop = 1
    else:
        stop = stop
    #使用jieba分词
    specialwords = "D:\\学习\\自然主义\\word2vec python包\\Semantic-Toolbox-main\\Semantic-Toolbox-main\\sempsy-1.0\\textprocess\\tests\\specialwords.txt"
    jieba.load_userdict(specialwords)
    words_cut = jieba.lcut(text.strip('\n'))
    if stop == 0:
        words = [w for w in words_cut if (w >= u'\u4e00' and w <= u'\u9fa5' and ' ' not in w)]
    else:
        # 保留所有中文字词，去除停用词
        words = [w for w in words_cut if (w >= u'\u4e00' and w <= u'\u9fa5' and w not in stopwords)]
    count = len(words)
    # print('分词结果：', '\n', words)
    # print('词数统计：', count)
    return words

def readfile(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        text = f.read().strip('\n')	
    return text

filepath = "D:\\学习\\自然主义\\语料库\\LIWC字典\\LIWC2015 Dictionary - Chinese (Simplified).dicx"
#CN_LIWC = read_dic(filepath)

with open(filepath, 'r', encoding='utf-8') as f:
    df = pd.read_csv(filepath, header=0, index_col='DicTerm')
print(df.columns)


storypath = "D:\\学习\\自然主义\\做语义网络\\分词\\STORY\\selected_sub\\story1txt"
#cut=[]
for file in os.listdir(storypath):
    file_name=os.path.abspath(os.path.join(storypath,file))
    #i=i+1
    data = readfile(file_name).strip()
    cut_words = textcut(data)
    #cut.append(cut_words)
    count = 0
    words = []
    for word in cut_words:
        if word in list(df.index):
            count += 1
            if df.loc[word, 'negate'] == 'X':
                words.append(word)
    try:
        #print(f'{os.path.basename(file).split(".")[0]} : {len(words)/len(sep_words)}')
        print(len(words)/len(cut_words))
    except ZeroDivisionError as e:
        print('0')

















