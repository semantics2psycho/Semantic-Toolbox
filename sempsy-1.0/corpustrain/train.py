
"""

Dependency:  https://pypi.org/project/gensim/

"""

# -*- coding: utf-8 -*-
import logging
from gensim.models import word2vec

#训练日志显示
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#输入训练的语料库
sentences = word2vec.Text8Corpus('baidu_wiki_corpus_cut_new-can.txt')  # 加载语料
#inp = open('para.seg.txt',encoding='utf-8')
#sentences = word2vec.LineSentence(inp)

#模型训练
#model = word2vec.Word2Vec(sentences, size=200, min_count=5,sg=1)  # 训练skip-gram模型; 默认window=5
model = word2vec.Word2Vec(sentences, size=300, window=10, min_count=5)   #训练CBOW模型，维度300
model.init_sims(replace=True)  #init_sims将使得模型的存储更加高效

# 保存模型，以便重用
model.save("E:\\WorkData\\NLP\Corpus\\model_zhwiki_wind10_300d_20210621.model")

# 以一种C语言可以解析的形式存储词向量
model.wv.save_word2vec_format("E:\\WorkData\\NLP\Corpus\\model_zhwiki_wind10_300d_20210621.bin", binary=True)
