"""
Module Help:
1.
2.
3.
4.
5.
6.

Dependency:
"""

from semdistance.distance import *

#主函数
if __name__ == '__main__':

    #定义停用词(可以手动设置，也有默认值)
    # stop_words = open('cn_stopwords.txt',encoding = 'utf-8').read().split('\n')

    #词向量模型(可以手动设置，也有默认值)
    # model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join('model_baiduwiki_sg_300d.bin'), binary=True)

    #测试文本
    test = open('testfile.txt','r',encoding='utf-8').read().strip('\n')
    test2 = open('testfile2.txt', 'r', encoding='utf-8').read().strip('\n')

    #分词
    word = textcut(test)
    print('分词结果：','\n',word)
    print('词数统计：',len(word))

    #计算句子相似性
    print('测试句子的语义相似性：',sim_sentence('他讨厌睡觉','我喜欢吃饭'))

    #计算两文本相似性
    print('测试文本的语义相似性：',sim_text(test,test2))

    #计算两个单词之间的相似性
    # w1 = '香蕉'
    # w2 = '苹果'
    # s = similarity(w1,w2)
    # d = 1-s
    # print(w1,' 和 ',w2,'的语义相似性为',s)
    # print(w1,' 和 ',w2,'的语义距离为',d)

    #也可以设置为输入任意两个单词或者句子或者文章计算语义距离
    print("------任意两个单词的相似性计算------")
    wt1 = input("请输入单词1：")
    wt2 = input("请输入单词2：")
    print(wt1,' 和 ',wt2,'的语义相似性为',similarity(wt1,wt2))

    #计算语义距离
    wind1, wind10, wind20 = text_distance_sentence(test, 1), text_distance_sentence(test, 10), \
                            text_distance_sentence(test, 20)

    print('基于单词的平均语义距离：',text_distance_word(test)[0])
    print('------不重叠句子窗口计算------')
    print('基于句子的平均语义距离wind1：', wind1[0])  # 句子窗口长度为1
    print('基于句子的平均语义距离wind10：',wind10[0]) #句子窗口长度为10
    # print('基于句子的平均语义距离wind15：', text_distance_sentence(test, 15))[0]  # 句子窗口长度为15
    print('基于句子的平均语义距离wind20：', wind20[0])  # 句子窗口长度为20
    print('------重叠滑动句子窗口计算------')
    print('基于句子的平均语义距离wind10：', distance_window(test, 10,1))  # 句子窗口长度为10,k=1
    print('基于句子的平均语义距离wind15：', distance_window(test, 15,1))  # 句子窗口长度为10,k=1
    print('基于句子的平均语义距离wind20：', distance_window(test, 20,1))  # 句子窗口长度为10,k=1
    print('------step=5，计算评价语义距离------')
    print('基于句子的平均语义距离wind15step5：', distance_window(test, 15, 5))  # 句子窗口长度为15,k=5
    print('基于句子的平均语义距离wind20step5：', distance_window(test, 20, 5))  # 句子窗口长度为20,k=5
    print('基于句子的平均语义距离wind30step5：', distance_window(test, 30, 5))  # 句子窗口长度为30,k=5