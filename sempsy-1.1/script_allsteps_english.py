# -*- coding: utf-8 -*-
"""
This script includes all the functional examples:
1. text process
2. vector training
3. semantic distance calculation
4. clustering
5. Classical paradigm data calculation

"""

"""
-----------------------------1. text process-----------------------------

"""
import pandas as pd
from textprocess.textcut import *

"part1. read file contents"
file = 'test/example/cut_create18.txt' # 单词数据
file = 'test/example/story.txt' # 长文本数据
#指定要读取的文件路径，这里的文件路径是test/testfile.txt。
content = readfile(file) 
content = readfiles(file,filetype='.txt',iflist='y')
#调用readfile函数读取文件内容，并将结果存储在变量content中。
#调用readfiles函数读取文件内容，并将结果存储在变量content中。
print('content：','\n',content)
#打印字符串"content："，然后换行，再打印变量content的值。

"part2. Character statistics"
charslen_default = charscount(content)
#调用charscount函数统计文本中的字符数量，并将结果存储在变量charslen_default中。默认情况下，该函数将包括标点符号在内的所有字符都计算在内。
# charslen_withpunc = charscount(content,0)
charslen_nopunc = charscount(content,1)
#调用charscount函数统计文本中去除标点符号后的字符数量，并将结果存储在变量charslen_nopunc中。通过传递参数1给函数，指示函数在统计字符数量时排除标点符号。
print('Word count with punctuation:',charslen_default)
#打印带有标点符号的字符数量的结果。
print('Word count no punctuation:',charslen_nopunc)
#打印去除标点符号后的字符数量的结果。

"part3. textcut"
"part3.A add specialwords"
special_words = open(os.path.join(r'E:\Semantic\Story\calculation\textcut\specialwords.txt'), encoding ='utf-8').read().split('\n')
#打开特殊词文件（specialwords.txt），读取其中的内容，并使用换行符(\n)将内容分割成一个词列表。特殊词列表用于扩展分词工具的词库。
jieba.load_userdict(special_words)
#使用jieba分词工具的load_userdict方法加载特殊词列表，将特殊词加入到分词工具的词库中，以便分词时能够识别这些特殊词。
"part3.B text cut"
words = textcut(content)
#调用textcut函数对文本进行分词，默认情况下不去除停用词，将分词结果存储在words变量中。
words_nonstop = textcut(content,0)
#调用textcut函数对文本进行分词，同时去除停用词，将分词结果存储在words_nonstop变量中。
# Set the stop word
# textcut(content,0,stopwords='D:/textprocess/tests/cn_stopwords.txt')
print('\ntext cut result with stopwords:\n',' '.join(words)) 
#打印带有停用词的分词结果。
print('\ntext cut result no stopwords:\n',' '.join(words_nonstop)) 
#打印不带停用词的分词结果。

"part4. part-of-speech tagging"
[part,partcount] = pos(puncdel(content))
#调用puncdel函数去除文本中的标点符号，然后调用pos函数对处理后的文本进行词性标注。返回的结果包括词性标注结果part和词性统计结果partcount，使用列表解构将其分别赋值给part和partcount两个变量。
print('part-of-speech tagging result：','\n',part)
#打印词性标注结果part，显示在屏幕上。
print('pos count：\n',partcount)
#打印词性统计结果partcount，显示在屏幕上。

"part5. word frequency statistics"
counts = wordcount(words)
#调用wordcount函数对分词结果words进行词频统计，将结果存储在变量counts中。
print('word count：\n',counts)
#打印词频统计结果counts，显示在屏幕上。
# display by row
# for x in counts:  
#     print(x[0],x[1])
t = type(counts)
#获取变量counts的数据类型。

"part6. Save the results to a file"
savefile(counts,'test/out.txt','.txt')
#调用savefile函数，将词频统计结果counts保存到文本文件out.txt中。文件路径为test/out.txt，文件格式为.txt。
savefile(counts,'test/out.csv','.csv')
#调用savefile函数，将词频统计结果counts保存到CSV文件out.csv中。文件路径为test/out.csv，文件格式为.csv。
savefile(counts,'test/out.xlsx','.xlsx')
#调用savefile函数，将词频统计结果counts保存到Excel文件out.xlsx中。文件路径为test/out.xlsx，文件格式为.xlsx。


"""
-----------------------------2. vector training-----------------------------

"""
from corpustrain.train import *

"part1. corpus clean"
corpusfile = 'test/testcorpus.txt'
corpus_content = readfile(corpusfile)
[words,count] = textcut(corpus_content)

"part2. corpus save"
outfile = 'corpus_text_cut.txt'
fd = open(outfile,'w',encoding=('utf8'))
fd.write(' '.join(words))
fd.close()

"part3. corpus training"
vector_train(outfile)
# default:
# vector_train(outfile,size_num=300,window_num=10,min_count_num=5,sg_num=0)
# out: test/word_vector_win10_300d.bin


"""
------------------------3. semantic distance calculation---------------------

"""
from semdistance.distance import *

"part1. distance calculation between words"
w1 = '父亲'
w2 = '爸爸'
dis_w = dis_words(w1,w2,1)
dis_E = dis_Euclidean_words(w1,w2,1)
print('the distance between ',w1,' and ',w2,' is ',dis_w)

"part2. distance calculation between word pairs"
pair1 = ['国王','男人']
pair2 = ['王后','女人']
dis_p = dis_pairs(pair1, pair2,1)
print('the distance between ',pair1,' and ',pair2,' is ',dis_p)

"part3. distance calculation between sentences"
s1 = '今天是晴天，我心情很好'
s2 = '美好的一天从精力充沛的早上开始'
"part3.A input format is wordcut"
dis_sentences(s1, s2,1,1)
"part3.B input format is text"
dis_sentences(s1, s2,1)

"part4. text distance calculation"
"part4.1 base word"
file = 'test/example/story.txt'
text = readfile(file)
text_dis = text_distance(text,1)

"part4.2 wind = 8"
text_dis_win = text_distance(text,1,8)

"part4.3 wind = 8,k = 5"
text_dis_win_k = text_distance(text,1,8,5)

"""
--------------------------------4. clustering--------------------------------

"""
"part1. k-means"
from cluster.kmeans import *
"part1.1 get word clustering labels"
# word_list : list of word in text
classCollect = {}
[classCollect,wordvector,clf] = word_classCollects(text_list,25)

"part1.2 visual word vector by clustering labels"
visual_cluster(text_list,25)

"part1.3 get the best K value"
get_K(wordvector)

"part1.4 get the sort of distance to cluster center"
dict_sort = {}
dict_sort = sortDisToClusterCenter(text_list,25)

"part2. tfidf"
from cluster.tfidf import *
from semdistance.get_vector import *
# # corpus split
# text = readfile('test/testfile.txt')
# # corpus split by '。' and textcut
# corpus = sentence_split_by_period(text)
# # corpus split by word length and textcut
# corpus = sentence_split_by_wordLength(text,50)
# # corpus split by document
# document_path = ''
# corpus = setence_split_by_document(document_path)
filepath = r'E:\Semantic\Story\calculation\textcut\cut\story1'
"part2.1 word tf-idf (sklearn)"
[corpus,word_tfidf] = word_tfidf_sklearn(filepath,'wordLength',50)
[corpus,word_tfidf] = word_tfidf_sklearn(filepath,'document')

"part2.2 word tf-idf (gensim)"
[corpus,word_tfidf] = word_tfidf_gensim(filepath,'wordLength',50)
[corpus,word_tfidf] = word_tfidf_gensim(filepath,'document')

"part2.3 tf-idf weighted vector"
weighted_vector = get_tfidf_weighted_vector(filepath,'gensim','document',1)

"""
-------------------5. Classical paradigm data calculation--------------------

"""

"part1. Forward flow test data calculation"
file = os.path.join('test/data_FFT.csv')
df = pd.read_csv(file)

key = df.values[:,0].tolist()
value = df.values[:,1:11].tolist()

FFT_distance = {}
for i in range(len(key)):
    FFT_distance[key[i]] = dis_FFT(value[i],1)

outfile = os.path.join('test/FFT_distance.csv')
FFT_df = pd.DataFrame.from_dict(FFT_distance, orient='index',columns=['distance'])
# FFT_df = FFT_df.reset_index().rename(columns = {'index':'subject'})
FFT_df.to_csv(outfile,sep=',')





