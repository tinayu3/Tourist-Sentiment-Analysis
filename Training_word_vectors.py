import jieba
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
import joblib
from sklearn.svm import SVC

# Read two categories of corpus
# pos = pd.read_excel('../data/travelTopic/gz_pos.xlsx', header=None)
# neg = pd.read_excel('../data/travelTopic/gz_neg.xlsx', header=None)
pos = pd.read_csv('../cleanData/data_yn_pos.txt', header=None)
neg = pd.read_csv('../cleanData/data_yn_neg.txt',  header=None)

# Perform word segmentation
pos['words'] = pos[0].apply(lambda x: jieba.lcut(x))
neg['words'] = neg[0].apply(lambda x: jieba.lcut(x))

# Merge the positive and negative corpora into training corpora and label them. The positive corpus is labeled 1 and the negative corpus is labeled 0.
x = np.concatenate((pos['words'], neg['words']))
#y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

#vector_size dimension
word2vec = Word2Vec(x, vector_size=200, window=3, min_count=5, sg=1, hs=1, workers=25)
word2vec.save('../model/word2vec_yn_200.model')
word2vec_model = Word2Vec.load('../model/word2vec_yn_200.model')
word2vec_model .wv.save_word2vec_format('../word2vecWordVector/word2vec_yn_200.txt', binary=False)