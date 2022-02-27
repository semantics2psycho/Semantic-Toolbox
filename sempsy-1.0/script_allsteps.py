# -*- coding: utf-8 -*-
"""
此脚本包括所有的功能示例
1. 文本分词
2. 词向量训练
3. 语义距离计算
"""

"""
1. 文本分词示例
"""

import pandas as pd
from textprocess.textcut import *

"0.输入你要分析的文件路径（绝对路径 或者 当前工作目录路径）"
file = 'testfile.txt'

"1.readfile(filepath)读取文件内容"
content = readfile(file)
print('文本内容为：','\n',content)

"2.charslen(text,punc)字符数统计"
charslen_default = charscount(content)
# charslen_nopunc = charscount(content,0)  # 等价于 charscount(content)
charslen_withpunc = charscount(content,1)
print('原文件字数统计：',charslen_default)
print('去除标点符号的字数统计：',charslen_withpunc)

"3.textcut(text,stop,stopwords)文本分词,stopwords可以自己设置，如不设置则使用默认的中文停用词表"
[words,count] = textcut(content) #等价于 textcut(content)
[words_nonstop,count_nonstop] = textcut(content,0)
# textcut(content,0,stopwords='D:/A-Data/PycharmProjects/sempsy/textprocess/tests/stop.txt')

print('\n')
print('删除停用词的分词结果:','\n',' '.join(words)) #词语之间以空格形式显示
print('词数统计：',count)
print('\n','不删除停用词的分词结果:','\n',' '.join(words_nonstop)) #词语之间以空格形式显示
print('词数统计：',count_nonstop)

"4.pos(text):词性标注"
[part,partcount] = pos(puncdel(content))
print('词性标注：','\n',part)
print('词性统计：','\n',partcount)

"5.wordcount(list):词频统计"
counts = wordcount(words)
print('词频统计：','\n',counts)
# for x in counts:  #按行显示
#     print(x[0],x[1])

"6.savefile():将结果保存成文件"
savefile(counts,'out.txt','.txt')
savefile(counts,'out.csv','.csv')
savefile(counts,'out.xlsx','.xlsx')
