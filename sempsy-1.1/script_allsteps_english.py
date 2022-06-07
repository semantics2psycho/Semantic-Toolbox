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
file = 'test/testfile.txt'
content = readfile(file)
print('content：','\n',content)

"part2. Character statistics"
charslen_default = charscount(content)
# charslen_withpunc = charscount(content,0)
charslen_nopunc = charscount(content,1)
print('Word count with punctuation:',charslen_default)
print('Word count no punctuation:',charslen_nopunc)

"part3. textcut"
"part3.A add specialwords"
special_words = open(os.path.join(r'E:\Semantic\Story\calculation\textcut\specialwords.txt'), encoding ='utf-8').read().split('\n')
jieba.load_userdict(special_words)
"part3.B text cut"
words = textcut(content)
words_nonstop = textcut(content,0)
# Set the stop word
# textcut(content,0,stopwords='D:/textprocess/tests/cn_stopwords.txt')
print('\ntext cut result with stopwords:\n',' '.join(words)) 
print('\ntext cut result no stopwords:\n',' '.join(words_nonstop)) 

"part4. part-of-speech tagging"
[part,partcount] = pos(puncdel(content))
print('part-of-speech tagging result：','\n',part)
print('pos count：\n',partcount)

"part5. word frequency statistics"
counts = wordcount(words)
print('word count：\n',counts)
# display by row
# for x in counts:  
#     print(x[0],x[1])
t = type(counts)
"part6. Save the results to a file"
savefile(counts,'test/out.txt','.txt')
savefile(counts,'test/out.csv','.csv')
savefile(counts,'test/out.xlsx','.xlsx')


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
file = 'test/testfile.txt'
text = readfile(file)
text_dis = text_distance(text)

"part4.2 wind = 8"
text_dis_win = text_distance(text,8)

"part4.3 wind = 8,k = 5"
text_dis_win_k = text_distance(text,8,5)

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





