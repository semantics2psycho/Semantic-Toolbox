# -*- coding: utf-8 -*-
"""
Created on Sat May  7 17:05:15 2022

@author: Dell
"""
from semdistance.distance import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


def clustr_kmeans(wordvector,classCount=30):

    clf = KMeans(n_clusters=classCount)
    s = clf.fit(wordvector)
    labels = clf.labels_
    
    return clf,labels

def get_K(wordvector):
    a = []
    
    for i in range(10,50):
        clf = KMeans(n_clusters=i)
        clf.fit(wordvector)
        a.append(clf.inertia_)
    
    X = range(10,50)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, a, 'o-')
    plt.show()

def word_classCollects(text_list,classCount=25,unique=0):
    wordvector = []
    for word in text_list:
        wordvector.append(get_word_vector(word,1))
    clf,labels = clustr_kmeans(wordvector,classCount)
    
    classCollects={}
    for i in range(len(wordvector)):
        if labels[i] in classCollects.keys():
            classCollects[labels[i]].append(text_list[i])
        else:
            classCollects[labels[i]] = [text_list[i]]
            
    return classCollects,wordvector,clf
    
# cents = clf.cluster_centers_
# inertia = clf.inertia_

def visual_cluster(text_list,classCount=25):

    pca = PCA(n_components=2)
    _,wordvector,clf = word_classCollects(text_list,classCount)
    vector_pca = pca.fit_transform(wordvector)
    vector_km = clf.fit_predict(vector_pca)
    
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.scatter(vector_pca[:,0],vector_pca[:,1],c=vector_km)
    
    for i in range(len(text_list)):    #给每个点进行标注
        plt.annotate(s=text_list[i], xy=(vector_pca[:, 0][i], vector_pca[:, 1][i]),
                     xytext=(vector_pca[:, 0][i] + 0.1, vector_pca[:, 1][i] + 0.1))
    plt.show()    

def sortDisToClusterCenter(text_list,classCount=25):
    
    _,wordvector,clf = word_classCollects(text_list,classCount)
    labels = clf.labels_.tolist()
    centers = clf.cluster_centers_
    
    num = 0
    dists = {}
    labels_dict = {}
    label_word_dis = {}
    # v = wordvector[0]
    for v in wordvector:
        center = centers[labels[num]]
        dist = dis_words_by_vec(v,center)
        dists[text_list[num]] = dist
        labels_dict[text_list[num]] = labels[num]
        num = num + 1
        
    # df = pd.DataFrame.from_dict(dists,orient='index',columns=['dis'])
    # df['label'] = labels_dict.values()
    
    for i in range(classCount):
        dis = {}
        for w,l in labels_dict.items():
            if l == i:
                dis[w] = dists[w]
        label_word_dis[i] = dis
    
    dis_sort_list = []
    for key,value in label_word_dis.items():
        dis_sort = sorted(value.items(),key=lambda x: x[1])
        dis_sort_list.append(dis_sort)
    
    # df.index[i]
    # df.values[:,1]
    # df1.dtypes
    
    return dis_sort_list


    

