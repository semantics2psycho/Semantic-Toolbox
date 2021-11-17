
"""
Module Help:
This is text preprocess module. Incluing 6 functions
1.readfile(): Read file content
2.puncdel():
3.charscount():
4.textcut():
5.pos():
6.wordcount():
7.savefile():


Dependency:  https://pypi.org/project/jieba/
             https://pandas.pydata.org/
"""

import os,re,string
import jieba
import jieba.posseg
import pandas as pd

"""Set stopwords: 
you can remove some meanningless and repetive words,which are stopwords.
for example, 的 了 呀 啊 呢 呵 , you can also set the punctuation into the stopwords.
"""
stop_words = open(os.path.abspath('../cn_stopwords.txt'), encoding ='utf-8').read().split('\n')

def readfile(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        text = f.read().strip('\n').replace(' ','')
        # print('原文件内容为：','\n',text)
    return text


#delete punctuation
def puncdel(text):
    punc_en = string.punctuation  # English Punctuation
    punc_cn = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）\u3000——+-=“：’；、。，？》《{}”' # Chinese Punctuation
    res = re.sub('[{}]'.format(punc_en),"",text)
    res = re.sub('[{}]'.format(punc_cn),"",res)
    print('去除标点符号后的文本：','\n',res)
    return res

#文本字符数统计,punc=1 则需要去除标点符号，反之则需统计标点符号，默认值punc=0
def charscount(text,punc=0):
    if punc == 1:
        wordall = puncdel(text).rstrip()
    else:
        wordall = text.rstrip()
    charslen = len(wordall)
    # print('字数统计：',charslen)
    return charslen


#分词统计，stop=1 则需要删除停用词，stop=0 则不删除停用词，默认值stop=1
def textcut(text,stop=1,stopwords=stop_words):
    if stop is None:
        stop = 1
    else:
        stop = stop
    #使用jieba分词
    words_cut = jieba.lcut(text.strip('\n'))
    if stop == 0:
        words = [w for w in words_cut if (w >= u'\u4e00' and w <= u'\u9fa5' and ' ' not in w)]
    else:
        # 保留所有中文字词，去除停用词
        words = [w for w in words_cut if (w >= u'\u4e00' and w <= u'\u9fa5' and w not in stopwords)]
    count = len(words)
    # print('分词结果：', '\n', words)
    # print('词数统计：', count)
    return words,count

def pos(text):
    words = jieba.posseg.lcut(text)
    words_pos = [ (w.word,w.flag) for w in words if w!=' ']
    posx = [w.flag for w in words if w!=' ']
    poscount = wordcount(posx)
    return words_pos,poscount

def wordcount(list):
    # count = Counter(words)  # 统计词频，词典元素为：{单词：次数}。
    # count_sort = sorted(dict(count).items(), key=lambda d: d[1], reverse=True)
    words = list
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    # 对词组进行排序
    count_sort = sorted(counts.items(),key=lambda x: x[1], reverse=True)
    return count_sort

def savefile(data,outfile,filetype):
    df = pd.DataFrame(data)
    if filetype=='.txt':
        df.to_csv(outfile,encoding='utf-8',index=False,header=False)
    elif filetype=='.csv':
        df.to_csv(outfile, encoding='utf_8_sig',index=False, header=False)
    elif filetype == '.xlsx':
        df.to_excel(outfile,encoding='utf-8',index=False,header=False)
    else:
        print('please input the right filetype!')






