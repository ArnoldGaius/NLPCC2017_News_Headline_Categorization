#coding:utf-8
#!/usr/bin/env python2
import pandas as pd
import datetime
import jieba 
import jieba.analyse

print '分词系统开始运行...'
start_time = datetime.datetime.now()
train_data_path = '../data/train.txt'
cut_train_data_path = '../data/cut_train.txt'
dev_data_path = '../data/dev.txt'
cut_dev_data_path = '../data/cut_dev.txt'
test_data_path = '../data/test.word'
cut_test_data_path = '../data/cut_test_data.txt'

jieba.load_userdict('../data/user_dict.txt') #装载用户词典
jieba.analyse.set_stop_words('../data/stopwords.txt')
train_data = pd.read_table(test_data_path,header=None,encoding='utf-8',names=['News'])

# print train_data.head()
print '分词开始...'
for i in range(len(train_data['News'])):
    News= ' '.join(jieba.analyse.extract_tags(train_data['News'][i], topK = 12, withWeight = False, allowPOS = ()))
    if len(News)>0:
        train_data['News'][i] = News
    else:
        print train_data['News'][i]
        train_data['News'][i] = ' '.join(jieba.cut(train_data['News'][i],cut_all = False))
        print train_data['News'][i]
print '分词结束'
# print train_data.head(10)
# train_data.dropna(axis=0)
print '数据存储中...'
pd.DataFrame(train_data).to_csv(cut_test_data_path,encoding='utf-8',header=None,index=False)
print '数据存储结束'
end_time = datetime.datetime.now()
print '运行耗时:\t'+str((end_time-start_time).seconds)+'s'
print '分词系统运行结束'
