import re
import logging
import pandas as pd
import jieba as jb
import warnings

warnings.filterwarnings("ignore")  # Ignore version issues

inFilePath='../originalData/sc_pos.xlsx'
outFilePath='../cleanData/data_sc_pos.txt'
dataset=pd.read_excel(inFilePath,header=None) #Input File
a=0
##Regularization to remove punctuation
r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[．.!/_,$&%^*()<>+""'?@|:~{}#]+|[—— - ！\\\，,。=？;←→、；⛲ ➕ ☺ ㉨ ● ☔ ✌ ⛴❗ ❤ ⏱ ⛰ ♥ ⛪ ⛅☀：★ ❣ … －－－ ～ ---「」 ⋯“”‘'￥……（）丶·^ * % @ 《》【】⭐ a-zA-Z0-9]"
#Stores the index value of the removed comments
listt=[]
delKeyWord= ['Wechat', 'Carpool', 'wx','Team up','Peers']
for i in range(len(dataset)):#Read line by line

    StrComment=dataset[0][i]
    logging.info('Delete comments that are not strings and are less than 5')
    if type(StrComment)!=str or len(StrComment)<=5:
        listt.append(i)
        continue
    logging.info('Delete reviews containing keywords')
    for j in delKeyWord:
        searchStr=StrComment.find(j)
        if searchStr != -1:
            listt.append(i)
            break
    ##Remove r4 regular expression filtering
    StrClean=re.sub(r4,'',StrComment)
    ##Apply regularization again to remove reviews other than text
    StrClean_1= re.sub(u"([^\u4e00-\u9fa5])", "", StrClean)
    ##Filter emojis by unicode
    try:
        res = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        res = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    dataFinalClean=res.sub(u'',StrClean_1)
    ##Re-assign rating
    dataset[0][i]=dataFinalClean
    #Check again whether the character length is less than 5
    if type(dataFinalClean)==float or len(dataFinalClean)<=5:
        listt.append(i)
    a+=1
    print(a)

dataset1=dataset.drop(listt,axis=0)
#dataset1.to_excel(inFilePath) #Output after cleaning


#################################################
f=open('../utilTxt/Stop_word_list.txt', 'r')#Stop Punctuation Dictionary
data=open(outFilePath, 'w')#
##Read the stop word list
stopDict=f.read()
#Word segmentation and dictionary loading
##Custom dictionary
jb.load_userdict("../utilTxt/Sentiment_Dictionary.txt")
# dataset=pd.read_excel(inFilePath)
dataCommet=dataset[0]
for k in range(len(dataCommet)):
    words=jb.lcut(dataCommet[k],cut_all=False)
    #Remove stop words
    for i in words:
        if i in stopDict:
            words.remove(i)

    #Remove duplicate words
    argWord=list(set(words))
    argWord.sort(key=words.index)
    lenarglist = len(argWord)
    # print(lenarglist)
    #Extract evaluation data and leave one space for each word
    for z in range(lenarglist):
        ##The lenarglist will be reduced by 1 when it is put into for
        # print("l", argWord[z])
        if z<lenarglist-1:
            data.writelines(argWord[z]+' ')
        elif z==lenarglist-1:
            data.writelines(argWord[z]+'\n')
    print(k)
