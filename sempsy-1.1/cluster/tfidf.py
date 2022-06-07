# -*- coding: utf-8 -*-
"""
1. merge_text
2. sentence_split_by_period
3. sentence_split_by_wordLength
4. sentence_list_split_by_wordLength
5. sentence_split_by_document
6. sentence_list_split_by_document
7. word_tfidf_sklearn
8. word_tfidf_gensim

"""

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer

import pandas as pd
import re,string
from textprocess.textcut import *
from gensim import corpora, models, similarities

# special_words = open(os.path.join('E:/Semantic/Story/calculation/textcut/specialwords.txt'), encoding ='utf-8').read().split('\n')
baiduwiki_corpus = 'H:/Corpus/corpus-nostop-cut/baidu_wiki_corpus_cut_new-can.txt'
# filepath = 'E:/Semantic/Story/calculation/textcut/story1txt'
       

def sentence_split_by_period(text):
    """
    Parameters
    ----------
    text : str
        corpus text
    
    Return
    ----------
    corpus : list-str
        Break up text into sentences with "。", save the textcut sentence as a list
    
    """
    punc_en = string.punctuation
    punc_cn = '~`!#$%^&*()_+-=|\';:/.,?><~·！@#￥%……&*（）\u3000——+-=“：，"’；、》《{}”' # Chinese Punctuation
    text = re.sub('[{}]'.format(punc_en),"",text)
    text = re.sub('[{}]'.format(punc_cn),"",text)
    text_list = text.replace('\n','' ).replace(' ','' ).replace('？','。' ).split('。')
    sentence_list = [w for w in text_list if w!='']
    
    corpus = []
    for sentence in sentence_list: 
        s = str(textcut(sentence)).replace('[','').replace(']','').replace('\'','').replace(',','')
        corpus.append(s) 
    corpus = [w for w in corpus if w!='']
    
    return corpus

def sentence_split_by_wordLength(filepath,legnth=50):
    """
    Parameters
    ----------
    filepath : str
        corpus text path
    legnth : int
        text split cut length
    
    Return
    ----------
    corpus : list-str
        Break up text into sentences by word length, save the textcut sentence as a list
    
    """
    text,corpus = merge_text(filepath)
    sentence_list = []
    for chunk in readfile_inchunks(corpus):
        # sentence_list = []
        text_list = textcut(chunk)
        # i = 2
        item_list = []
        
        for i in range(0,len(text_list)):
            # index = i // 50
            flag = i % legnth
            if i == 0:
                item_list.append(text_list[i])
            else:
                if flag != 0:
                    item_list.append(text_list[i])
                else:
                    sentence_list.append(str(item_list).replace('[','').replace(']','').replace('\'','').replace(',',''))
                    item_list = []
                
        # sentences_list.append(sentence_list)      
    return sentence_list    

def sentence_list_split_by_wordLength(filepath,legnth=50):
    """
    Parameters
    ----------
    filepath : str
        corpus text path
    legnth : int
        text split cut length

    
    Return
    ----------
    corpus : list-str
        Break up text into sentences by word length, save the textcut sentence as a list
    
    """
    sentence_list = []

    for chunk in readfile_inchunks(corpus):
        # sentence_list = []
        text_list = textcut(chunk)
        # i = 2
        item_list = []
        
        for i in range(0,len(text_list)):
            # index = i // 50
            flag = i % legnth
            if i == 0:
                item_list.append(text_list[i])
            else:
                if flag != 0:
                    item_list.append(text_list[i])
                else:
                    sentence_list.append(item_list)
                    item_list = []
                
        # sentences_list.append(sentence_list)      
    return sentence_list    

        
def sentence_split_by_document(filepath):
    """
    Parameters
    ----------
    filepath : str
        document path
    
    Return
    ----------
    corpus : list-str
        Break up text into sentences by document, save the textcut sentence as a list
    
    """
    # filepath = 'E:/Semantic/Story/calculation/textcut/story1txt'	# 存放原始文件的路径

    list_name = []	# 存放所有原始数据的绝对路径
    file_name = []	# 存放所有原始文件的名字
    sentence_list = []  # 以字符串形式存放句子
    
    # file = os.listdir(filepath)[0]
    for file in os.listdir(filepath):
        words_list = []
        file_path = os.path.join(filepath,file)
        file_name = os.path.splitext(file)[0]
        
        content = readfile(file_path)
        words_list = textcut(content)
        
        sentence_list.append(str(words_list).replace('[','').replace(']','').replace('\'','').replace(',',''))
        
    return sentence_list
        
def sentence_list_split_by_document(filepath):
    """
    Parameters
    ----------
    filepath : str
        document path
    
    Return
    ----------
    corpus : list-list
        Break up text into sentences by document, save the textcut sentence as a list
    
    """
    # filepath = 'E:/Semantic/Story/calculation/textcut/story1txt'	# 存放原始文件的路径

    list_name = []	# 存放所有原始数据的绝对路径
    file_name = []	# 存放所有原始文件的名字
    sentence_list = []  # 以字符串形式存放句子
    
    # file = os.listdir(filepath)[0]
    for file in os.listdir(filepath):
        words_list = []
        file_path = os.path.join(filepath,file)
        file_name = os.path.splitext(file)[0]
        
        content = readfile(file_path)
        words_list = textcut(content)
        
        sentence_list.append(words_list)
        
    return sentence_list


def word_tfidf_sklearn(filepath,splitType,length=50):
    """
    Parameters
    ----------
    filepath : str
        text file path
    splitType : str
        'wordLength' : sentence split by wordLength
        'document' : sentence split by document
    legnth : int
        text split cut length

        
    Return
    ----------
    df_word_tf : Dataframe
        TF of word Result
    df_word_idf : Dataframe
        IDF of word Result
    df_word_tfidf : Dataframe
        TF-IDF of word Result
    """
    if splitType == 'wordLength':
        corpus = sentence_split_by_wordLength(filepath,length)
        corpus_list = sentence_list_split_by_wordLength(filepath,length)
    elif splitType == 'document':
        corpus = sentence_split_by_document(filepath)
        corpus_list = sentence_list_split_by_document(filepath)
    
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    count = vectorizer.fit_transform(corpus)
    word = vectorizer.get_feature_names()
    # df_word_tf =  pd.DataFrame(count.toarray(),columns=word) 
    transformer = TfidfTransformer(smooth_idf=True,norm='l2',use_idf=True)
    
    tfidf_matrix = transformer.fit_transform(count)
    tfidf_matrix_array = tfidf_matrix.toarray()
    word_tfidf = []
    
    for i in range(len(corpus)):
        tfidf_list = []
        m = []
        m = tfidf_matrix_array[i]
        tfidf_list = {word[i]:m[i] for i in range(len(m)) if m[i]!=0}
        word_tfidf.append(tfidf_list)  

    # df_word_tfidf = pd.DataFrame(tfidf_matrix.toarray(),columns=word)
    # df_word_idf = pd.DataFrame(list(zip(vectorizer.get_feature_names(),transformer.idf_)),columns=['单词','idf'])
    
    return corpus_list,word_tfidf

 
def word_tfidf_gensim(filepath,splitType,length=50):
    """
    Parameters
    ----------
    filepath : str
        text file path
    splitType : str
        'wordLength' : sentence split by wordLength
        'document' : sentence split by document
    legnth : int
        text split cut length
        
    Return
    ----------
    wordid : dict
        key : word name; value : word id
    word_tfidf : list - tuple(wordid,tfidf)
        word tfidf value by document/wordLength
    """
    
    if splitType == 'wordLength':
        corpus = sentence_list_split_by_wordLength(filepath,length)
    elif splitType == 'document':
        corpus = sentence_list_split_by_document(filepath)

    
    dictionary = corpora.Dictionary(corpus)  # 建立词典   
    wordid = dictionary.token2id
    idword = {vi:ki for ki,vi in wordid.items()}
    
    corpus2bow = [dictionary.doc2bow(doc) for doc in corpus]
    
    tfidf_model = models.TfidfModel(corpus2bow)
    tfidf_matrix_gensim = tfidf_model[corpus2bow]  # 得到语料的tfidf值
    word_tfidf = []
    # wtfidf = tfidf_matrix_gensim[0]
    for i in range(len(corpus)):
        tfidf_list = []
        tfidf_list = {idword[m[0]]:m[1] for m in tfidf_matrix_gensim[i]}
        word_tfidf.append(tfidf_list)  
    
    return corpus,word_tfidf


