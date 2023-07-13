
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
import sys,codecs
import importlib,sys
from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
importlib.reload(sys)

"""Set stopwords: 
you can remove some meanningless and repetive words,which are stopwords.
for example, 的 了 呀 啊 呢 呵 , you can also set the punctuation into the stopwords.
"""
stop_words = open(os.path.join('textprocess/tests/cn_stopwords.txt'), encoding ='utf-8').read().split('\n')
stop_words_en = stopwords.words('english')

def readfile(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        text = f.read().strip('\n')
        # print('原文件内容为：','\n',text)
    return text

def readfiles(filepath,filetype='.txt',iflist='y'):
    if filetype == '.txt':
        with open(filepath,'r',encoding='utf-8') as f:
            if iflist == 'y':
                text = f.read().strip('\n').split(' ')
                # print('原文件内容为：','\n',text)
            else :
                text = f.read().strip('\n')
                    
    elif filetype == '.xlsx':
        df = pd.read_excel(filepath)
        value = df.values[:,1:]
        text = value.tolist()
    return text

def readfile_inchunks(filepath,chunk_size=1024*1024):
    file_object = codecs.open(filepath,'r','utf-8')
    while True:
        chunk = file_object.read(chunk_size)
        if not chunk:
            break
        yield chunk
        

#delete punctuation and english
def puncdel(text):
    punc_en = string.punctuation  # English Punctuation
    punc_cn = 'a-zA-Z~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）\u3000——+-=“：’；、。，？》空《{}”' # Chinese Punctuation
    res = re.sub('[{}]'.format(punc_en),"",text)
    res = re.sub('[{}]'.format(punc_cn),"",res)
    print('text without Punctuation：','\n',res)
    return res

def puncdel_en(text):
    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*'
    res = re.sub('[{}]'.format(punc),"",text)
    print('text without Punctuation：','\n',res)
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
    return words

def textcut_en(text,stopwords=stop_words_en):
    text_del = puncdel_en(text)
    words_cut = text_del.split()
    words = [w for w in words_cut if w not in stopwords]
    return words

flag_en2cn = {
    'a': '形容词', 'ad': '副形词', 'ag': '形语素', 'an': '名形词', 'b': '区别词',
    'c': '连词', 'd': '副词', 'df': '不要', 'dg': '副语素','eng':'英文',
    'e': '叹词', 'f': '方位词', 'g': '语素', 'h': '前接成分',
    'i': '成语', 'j': '简称略语', 'k': '后接成分', 'l': '习用语',
    'm': '数词', 'mg': '数语素', 'mq': '数量词',
    'n': '名词', 'ng': '名语素', 'nr': '人名', 'nrfg': '古代人名', 'nrt': '音译人名',
    'ns': '地名', 'nt': '机构团体', 'nz': '其他专名',
    'o': '拟声词', 'p': '介词', 'q': '量词',
    'r': '代词', 'rg': '代语素', 'rr': '代词', 'rz': '代词',
    's': '处所词', 't': '时间词', 'tg': '时间语素',
    'u': '助词', 'ud': '得', 'ug': '过', 'uj': '的', 'ul': '了', 'uv': '地', 'uz': '着',
    'v': '动词', 'vd': '副动词', 'vg': '动语素', 'vi': '动词', 'vn': '名动词', 'vq': '动词',
    'x': '非语素字', 'y': '语气词', 'z': '状态词', 'zg': '状态语素',
}

def pos(text):
    words = jieba.posseg.lcut(text)
    words_pos = [(w.word,flag_en2cn[w.flag]) for w in words if w!=' ']
    posx = [flag_en2cn[w.flag] for w in words if w!=' ']
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

def savefile(data,outfile,filetype,index_in=0,header_in=0):
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        df = pd.DataFrame.from_dict(data.items())
    else:
        df = pd.DataFrame(data)
    
    if index_in == 0:
        index_in = False
    else:
        index_in = True
    if header_in == 0:
        header_in = False
    else:
        header_in = True
    
    if filetype=='.txt':
        df.to_csv(outfile,encoding='utf-8',index=index_in,header=header_in)    
    elif filetype=='.csv':
        df.to_csv(outfile, encoding='utf_8_sig',index=index_in, header=header_in)
    elif filetype == '.xlsx':
        df.to_excel(outfile,encoding='utf-8',index=index_in,header=header_in)
    else:
        print('please input the right filetype!')


def merge_text(filepath,outfile):
    """
    Parameters
    ----------
    filepath : str
        corpus text path
    legnth : int
        text split cut length
    
    Return
    ----------
    text : str
        text after cut
    outfile : str
        text out file path
    
    """
    # outfile = 'E:/Semantic/Story/calculation/textcut/merge_text.txt'
    
    list_name = []	# 存放所有原始数据的绝对路径
    # file_name = []	# 存放所有原始文件的名字
    text_list = []

    text_out = []
    # file = os.listdir(filepath)[0]
    
    for file in os.listdir(filepath):
        file_path = os.path.join(filepath,file)
        # file_name = os.path.splitext(file)[0]
        
        content = readfile(file_path)
        text_str = ' '.join(textcut(content))
        text_list.append(text_str)
        
    text = ' '.join(text_list)
    text_out.append(text)
    savefile(text_out,outfile,'.txt')
    
    return text,outfile




