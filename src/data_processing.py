#coding:utf-8
'''
Created on 2017年5月14日

@author: jiang
'''
from sklearn.naive_bayes import MultinomialNB
import ProVector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_score, KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import datetime
from sklearn.metrics.classification import confusion_matrix
from numpy import matrix

print '预测系统开始运行...'
start_time = datetime.datetime.now()#start time

train_data_path = '../data/cut_train.txt'
test_data_path = '../data/test_data.txt'
cut_train_data_path = '../data/cut_train.txt'
cut_dev_data_path = '../data/cut_dev.txt'
dev_data_path = '../data/dev.txt'

Synthetic_training_set = '../data/Synthetic_training_set.txt'


print '文件读取...'
#read_csv(File_name,hearder,coding_type,matrix_col_names)
train_data = pd.read_csv(cut_train_data_path,header=None,encoding='utf-8',names=['Categorization','News'])
test_data = pd.read_csv(cut_dev_data_path,header=None,encoding='utf-8',names=['Categorization','News'])
print '文件读取完成!'


X_train = train_data.News
X_test = train_data.News
Y_train = train_data.Categorization
Y_test = train_data.Categorization

def get_stop_words():
    result = set()
    for line in open('../data/stopwords.txt', 'r').readlines():
        result.add(line.strip())
    return result

print '构建tf_idf量化器...'
#构建包含量化器(vectorizers)和分类器
vectorizer = ProVector.TfidfPro_Vectorizer(stop_words=get_stop_words(),smooth_idf=True, sublinear_tf=True,use_idf=True,use_Wt=False,norm='l1')
vectorizer.fit_transform(X_train,Y_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)
# model = LogisticRegression(class_weight="balanced") #逻辑回归
model = MultinomialNB(alpha=0.1,fit_prior=False) #多项式贝叶斯


#评估分类器性能
print '构建预测模型...'
model.fit(X_train, Y_train)
# print "model feature count:"
# print model.feature_count_
print "Accuracy on training set:"
print model.score(X_train, Y_train)
y_predict = model.predict(X_test)
print "Classification Report:"
print metrics.classification_report(Y_test,y_predict)
# print "Confusion Matrix:"
# print metrics.confusion_matrix(Y_test,y_predict)


Y_test = test_data.Categorization
X_test = vectorizer.transform(test_data.News)
y_predict = model.predict(X_test)
y_predict = pd.Series(y_predict)

print "Accuracy on testing set:"
print model.score(X_test,Y_test)
print "Classification Report:"
print metrics.classification_report(Y_test,y_predict)


end_time = datetime.datetime.now()#start time
print '预测系统运行完成!'
print 'used time:\t'+str((end_time-start_time).seconds)+'s'

#设置pandas输出格式化
pd.set_option('display.height',1000)
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

#混淆矩阵
print 'Confusion Matrix :'
cm = metrics.confusion_matrix(Y_test,y_predict,labels=Y_test.unique())
print pd.DataFrame(cm,columns = Y_test.unique(),index = Y_test.unique())


print '图像显示...'
list_y_test = Y_test.value_counts().sort_index()
list_y_predict = y_predict.value_counts().sort_index()
test_predict_count_df = pd.concat([list_y_test,list_y_predict,list_y_predict-list_y_test,abs(list_y_test-list_y_predict)],axis=1,keys=['Test count:','Predict count:','Sub Result:','Sub_Abs Result:'])
print test_predict_count_df

fig, ax = plt.subplots()  
index = np.arange(len(Y_train.unique()))  
bar_width = 0.35
  
opacity = 0.4
rects1 = plt.bar(index, list_y_test, bar_width,alpha=opacity, color='b',label='test')  
rects2 = plt.bar(index + bar_width, list_y_predict, bar_width,alpha=opacity,color='r',label='predict') 
plt.xlabel('Group')
plt.ylabel('Value')
plt.title('Value by group')  
plt.xticks(index + bar_width, (list_y_predict.index),rotation=-30)  
plt.ylim(0,test_predict_count_df.values.max()*1.1)  
plt.legend()  

plt.tight_layout()  
plt.show()

pd.DataFrame(y_predict).to_csv('SGDClassifier_predict.txt',header=None,index=None,encoding='utf-8')

