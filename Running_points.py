from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import jieba as jb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import warnings

warnings.filterwarnings("ignore")  # Ignore version issues
def loadGLoveModel(filename):
    embeddings_index = {}
    f = open(filename, encoding='UTF-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def suc_train(train_vecs, y_train, test_vecs, y_test):
    # Creating the SVC Model
    print("#Creating the SVC Model")
    cls = SVC(kernel="rbf", verbose=True, shrinking=0)
    #Training the model#
    cls.fit(train_vecs, y_train)  # The training set data, the second is the training set label
    # Save the model
    joblib.dump(cls, "../model/svcmodel.pkl")
    # Output score
    print("SVC Rate:", cls.score(test_vecs, y_test))


def logistic_train(train_vecs, y_train, test_vecs, y_test):
    print("#Creating a Logistic Regression Model")
    #Training the model#
    regr = LogisticRegression()
    regr.fit(train_vecs, y_train)
    # Save the model
    joblib.dump(regr, "../model/logisticmodel.pkl")
    print("Logisitic Rate:", regr.score(test_vecs, y_test))
def naivenayesian_train(train_vecs, y_train, test_vecs, y_test):
    print("#Creating a Gaussian Naive Bayes Model")
    clf = GaussianNB()
    # Training with Naive Bayes
    clf.fit(train_vecs, y_train)
    # Save the model
    joblib.dump(clf, "../model/naivenayesianmodel.pkl")
    print("Gaussian Naive Bayes Scoring:", clf.score(test_vecs, y_test))

def SVM_PRF():
    print("#SVC model performance evaluation")
    train_vecs = np.load("../model/train_vecs.npy")
    regr = joblib.load("../model/svcmodel.pkl")
    y_pred = regr.predict(train_vecs)
    y_true = np.load("../model/y_train.npy")
    y_pred = y_pred.astype(np.int)
    y_true = y_true.astype(np.int)
    tp = sum(y_true & y_pred)  # result1
    fp = sum((y_true == 0) & (y_pred == 1))  # result1
    tn = sum((y_true == 0) & (y_pred == 0))  # result0
    fn = sum((y_true == 1) & (y_pred == 0))  # result2
    # print("tp", tp)
    # print("fp", fp)
    # print("tn", tn)
    # print("fn", fn)
    POS_P = tp / (tp + fp)
    POS_R = tp / (tp + fn)
    POS_F = (2 * POS_R * POS_P) / (POS_R + POS_P)
    NEG_P = tn / (tn + fn)
    NEG_R = tn / (tn + fp)
    NEG_F = (2 * NEG_R * NEG_P) / (NEG_R + NEG_P)
    # print("POS_P", POS_P)
    # print("POS_R", POS_R)
    # print("POS_F", POS_F)
    # print("NEG_P", NEG_P)
    # print("NEG_R", NEG_R)
    # print("NEG_F", NEG_F)
    print(POS_P)
    print(POS_R)
    print(POS_F)
    print(NEG_P)
    print(NEG_R)
    print(NEG_F)

def logistic_PRF():
    print("#Logistic regression model performance evaluation")
    train_vecs = np.load("../model/train_vecs.npy")
    regr = joblib.load("../model/logisticmodel.pkl")
    y_pred = regr.predict(train_vecs)
    y_true = np.load("../model/y_train.npy")
    y_pred = y_pred.astype(np.int)
    y_true = y_true.astype(np.int)
    tp = sum(y_true & y_pred)  # Result1
    fp = sum((y_true == 0) & (y_pred == 1))  # result1
    tn = sum((y_true == 0) & (y_pred == 0))  # result0
    fn = sum((y_true == 1) & (y_pred == 0))  # result2
    # print("tp", tp)
    # print("fp", fp)
    # print("tn", tn)
    # print("fn", fn)
    POS_P = tp / (tp + fp)
    POS_R = tp / (tp + fn)
    POS_F = (2 * POS_R * POS_P) / (POS_R + POS_P)
    NEG_P = tn / (tn + fn)
    NEG_R = tn / (tn + fp)
    NEG_F = (2 * NEG_R * NEG_P) / (NEG_R + NEG_P)
    # print("POS_P", POS_P)
    # print("POS_R", POS_R)
    # print("POS_F", POS_F)
    # print("NEG_P", NEG_P)
    # print("NEG_R", NEG_R)
    # print("NEG_F", NEG_F)
    print(POS_P)
    print(POS_R)
    print(POS_F)
    print(NEG_P)
    print(NEG_R)
    print(NEG_F)
def naivenayesian_PRF():
    print("#Gaussian Naive Bayes Model Performance Evaluation")
    train_vecs = np.load("../model/train_vecs.npy")
    regr = joblib.load("../model/naivenayesianmodel.pkl")
    y_pred = regr.predict(train_vecs)
    y_true = np.load("../model/y_train.npy")
    y_pred = y_pred.astype(np.int)
    y_true = y_true.astype(np.int)
    tp = sum(y_true & y_pred)  # Result1
    fp = sum((y_true == 0) & (y_pred == 1))  # Result1
    tn = sum((y_true == 0) & (y_pred == 0))  # Result0
    fn = sum((y_true == 1) & (y_pred == 0))  # Result2
    # print("tp", tp)
    # print("fp", fp)
    # print("tn", tn)
    # print("fn", fn)
    POS_P = tp / (tp + fp)
    POS_R = tp / (tp + fn)
    POS_F = (2 * POS_R * POS_P) / (POS_R + POS_P)
    NEG_P = tn / (tn + fn)
    NEG_R = tn / (tn + fp)
    NEG_F = (2 * NEG_R * NEG_P) / (NEG_R + NEG_P)
    # print("POS_P", POS_P)
    # print("POS_R", POS_R)
    # print("POS_F", POS_F)
    # print("NEG_P", NEG_P)
    # print("NEG_R", NEG_R)
    # print("NEG_F", NEG_F)
    print(POS_P)
    print(POS_R)
    print(POS_F)
    print(NEG_P)
    print(NEG_R)
    print(NEG_F)

def build_vector(text, size, wv):
    # Create a data space of a specified size
    vec = np.zeros(size).reshape((1, size))

    # Count is the number of word vectors counted
    count = 0
    # Loop through all word vectors and sum them up
    for w in text:
        try:
            vec += wv[w].reshape((1, size))
            count += 1
            # print(w)
        except:
            continue

    # After the loop is completed, calculate the average
    if count!=0:
        vec/=count
    return vec

def logisitic_predict(string):  # 测试
    # Segment the sentence
    words = jb.cut(string)
    # Convert word segmentation results into word vectors
    word_vecs = get_predict_vecs(words)
    # Loading the model
    regr = joblib.load("../model/logisticmodel.pkl")
    result = regr.predict(word_vecs)
    if result[0] == 1:
        print("pos")
    else:
        print("neg")


def svm_predict(string):  # test
    # Segment the sentence
    words = jb.cut(string)
    # Convert word segmentation results into word vectors
    word_vecs = get_predict_vecs(words)
    # Loading the model
    cls = joblib.load("../model/svcmodel.pkl")
    # Prediction results
    result = cls.predict(word_vecs)
    # Output
    if result[0] == 1:
        print("pos")
    else:
        print("neg")
def get_predict_vecs(words):
    # Loading word vector model
    print("#Loading word vector model")
    wv = Word2Vec.load("../model/word2vec.model")
    # Convert new words into vectors
    train_vecs = build_vector(words, 300, wv)
    return train_vecs
if __name__ == '__main__':
    print("Getting Started")
    #List, labelList = loadData()  # Loading corpus data
    neg = pd.read_excel("../originalData/yn_neg.xlsx", header=None)  # negative
    pos = pd.read_excel("../originalData/yn_pos.xlsx", header=None)  # positive
    # These are two types of data, all of which are x values.
    pos['words'] = pos[0].apply(lambda x: list(jb.cut(x)))
    neg['words'] = neg[0].apply(lambda x: list(jb.cut(x)))
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
    # Need y value 0 represents neg 1 represents pos
    X = np.concatenate((pos['words'], neg['words']))
    print("X-size:", len(X))
    # Array concatenation
    # Split into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
    np.save("../model/y_train.npy", y_train)
    np.save("../model/y_test.npy", y_test)
    # print(X_train)
    np.save("../model/x_train.npy", X_train)
    np.save("../model/x_test.npy", X_test)

    gloveModel =loadGLoveModel('../word2vecWordVector/word2vec_yn_300.txt')

    train_vecs = np.concatenate([build_vector(z, 300,gloveModel) for z in X_train])
    np.save('../model/train_vecs.npy', train_vecs)
    #print(train_vecs)
    test_vecs = np.concatenate([build_vector(z, 300,gloveModel) for z in X_test])
    np.save('../model/test_vecs.npy', test_vecs)
    suc_train(train_vecs, y_train, test_vecs, y_test)  # SVC
    logistic_train(train_vecs, y_train, test_vecs, y_test)  # logistic regression
    naivenayesian_train(train_vecs, y_train, test_vecs, y_test)  # Naive Bayes
    SVM_PRF()
    logistic_PRF()
    naivenayesian_PRF()