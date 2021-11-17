
import gensim
import numpy as np
from textprocess.textcut import *

#词向量模型
model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join('model_baiduwiki_sg_300d.bin'), binary=True)

# 获取词语向量
def get_vector(word):
    try:
        return model[word]
    except:
        print(word)
        print('该词语在词典中未检测到！')
        # return np.zeros(300)[: int(300)]

# norm计算向量的模；求单位向量
def _unitvec(v): return v/ np.linalg.norm(v)


# 求词语之间的相似性，即求向量的余弦值
def similarity(word1, word2, size=300):
    return np.dot(_unitvec(get_vector(word1)),
                  _unitvec(get_vector(word2)))


# #求句子平均向量
def mean_vector_sentence(s):
    vectorized_sentence = [get_vector(w) for w in textcut(s)]
    mean_vector_sentence = np.mean(vectorized_sentence, 0)
    return mean_vector_sentence


# 求句子相似性
def sim_sentence(s1, s2):
    return np.dot(_unitvec(mean_vector_sentence(s1)), _unitvec(mean_vector_sentence(s2)))


def mean_distances(vectors, vec_len):
    dists = []
    for i in range(vec_len):
        for j in range(vec_len):
            dists.append(1 - np.dot(_unitvec(vectors[i]), _unitvec(vectors[j])))
    mean_distance = np.mean(dists)

    return mean_distance


# 基于词语的文章平均语义距离
def text_distance_word(text):
    words = textcut(text)
    wc = len(words)  # 词语个数
    vecs_word = [get_vector(w) for w in words]
    mean_vec = np.mean(vecs_word, 0)  # 文本平均向量
    distance = mean_distances(vecs_word, len(vecs_word))  # 文章平均语义距离

    return distance, mean_vec


# 基于句子的文章平均语义距离,text为文本，wind为句子长度,句子间不重叠
def text_distance_sentence(text, wind):
    wcut = textcut(text)  # 分词
    # k = 5 #i的步长
    wc = len(wcut)  # 词语数
    left = wc % wind  # 余数
    # m = int(l / wind) * wind-k
    m = wc - left - 1  # 最后一个句子的索引值
    vec_sents = []  # 每个句子的向量
    for i in range(0, m + 1, wind):
        if (i != m):
            vec_sents.append(np.mean([get_vector(w) for w in wcut[i:i + wind]], 0))
        else:
            vec_sents.append(np.mean([get_vector(w) for w in wcut[i:]], 0))
    distance = mean_distances(vec_sents, len(vec_sents))  # 平均语义距离
    mean_vec = np.mean(vec_sents, 0)  # 文章平均向量
    return distance, mean_vec


# 计算全局语义距离，即文章的平均语义距离，滑动窗口，句子间有重叠，k为滑动距离
def distance_window(text, wind, k):
    words_cut = textcut(text)  # 分词
    wc = len(words_cut)
    # k = 10  # i的步长
    # c = int(words_len/wind)*wind-b
    c = int((wc - wind) / k) * k  # 最后一个句子的索引值
    vectors_sent = []
    # 所有句子窗口向量
    for i in range(0, c + 1, k):
        if (i != c):
            vectors_sent.append(np.mean([get_vector(w) for w in words_cut[i:i + wind]], 0))
        else:
            vectors_sent.append(np.mean([get_vector(w) for w in words_cut[i:]], 0))
    # vectors_sent = [np.mean([get_vector(w) for w in words_cut[i:i + wind]],0) for i in range(0,c,b)]

    sent_len = len(vectors_sent)  # 句子向量的个数
    dists_between_sents = []  # 文章中任意两个句子间的语义距离
    for i in range(sent_len - 1):
        for j in range(i + 1, sent_len):
            sim = np.dot(_unitvec(vectors_sent[i]), _unitvec(vectors_sent[j]))
            dists_between_sents.append(1 - sim)
    global_dist = np.mean(dists_between_sents)  # 文章平均语义距离

    return global_dist


# 求两文本相似性(以词语为单位)
def sim_text(text1, text2):
    return np.dot(text_distance_word(text1)[1], text_distance_word(text2)[1])