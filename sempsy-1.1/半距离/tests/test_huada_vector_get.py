# -*- coding: utf-8 -*-
"""
Created on Fri May 20 21:20:37 2022

@author: Dell
"""
import os
import pandas as pd
from textprocess.textcut import *
from semdistance.distance import *

def readfile(filepath):
    with open(filepath,'r',encoding='gbk') as f:
        text = f.read().strip('\n').replace(' ','')
        # print('原文件内容为：','\n',text)
    return text


path = 'E:/Semantic/Caculate/Distance/Huada/联想任务/联想流畅性健康'	# 存放原始文件的路径
outpath = 'E:/Semantic/Caculate/Distance/Huada/vector_file/联想流畅性健康'
list_name = []	# 存放所有原始数据的绝对路径
file_name = []	# 存放所有原始文件的名字

file = os.listdir(path)[0]
for file in os.listdir(path):
    file_path = os.path.join(path,file)
    file_name = os.path.splitext(file)[0]
    
    content = readfile(file_path)
    punc_cn = '空~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）\u3000——+-=“：’；、。，？》《{}”' # Chinese Punctuation
    content = re.sub('[{}]'.format(punc_cn),"",content)
    list_c = []
    list_c = content.split('\n')
    list_noE_c = [w for w in list_c if (w >= u'\u4e00' and w <= u'\u9fa5' and ' ' not in w)]

    vector_list = []
    vector_list = get_word_sequence_vector(list_noE_c,1)
            
    vector_list = [v for v in vector_list if v.size > 1]
    
    outfile = os.path.join(outpath,file)
    savefile(vector_list, outfile, '.txt')
    print(file_name)
    
    


