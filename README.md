# NLPCC2017项目2 新闻标题分类 NLPCC2017_Task2_News_Headline_Categorization 

题目描述 GuideLine
==============================
You Can get it [GuideLine](https://github.com/ArnoldGaius/NLPCC2017_Task2_News_Headline_Categorization/blob/master/GuideLine/taskgline02.pdf)

训练/测试集下载 Train/Test DataSets Download
==============================
You Can get it [Here](https://pan.baidu.com/s/1qXYzB5a) Click **下载** to Download it

**Train Data Format**

|   **type**  |                     **Text**                        |
|:-----------:|:---------------------------------------------------:|
|entertainment|   台媒预测周冬雨金马奖封后，大气的倪妮却佳作难出        |
|     food    |   农村就是好，能吃到纯天然无添加的野生蜂蜜，营养又健康   |
|    fashion  |   14款知性美装，时尚惊艳搁浅的阳光轻熟的优雅            |
|     etc.    |                       etc.                          |

解题思路 Solving ideas 
============================== 
- 文本Text去除停用词 cut Text and drop stop_words
- 使用Tf-idf计算词向量值，建立词向量表示矩阵 construct Word Vector Matrix by calculating word vector number using Tf-idf<br>
**It is worth mentioning that: if there are too many features, sparse matrix can help you save a lot of memory**
- 建立模型 creat Model <br>
**I tried many Models(svm、cnn、logisticRegression，etc.) on this task but `FINALLY` I chosed MultinomialNB just because it cost least time and got high precision recall and F1-score**
- 预测结果 predict

最终结果 Finally result
=============================
|   **Test/Train %**  | **avg-precision**| **avg-recall**| **avg-F1-score**| **time-cost(s)**|
|:-------------------:|:----------------:|:-------------:|:---------------:|:---------------:|
|          100%       |        93%       |       93%     |       93%       |        8s       |
|          75%        |        93%       |       93%     |       93%       |        7s       |
|          50%        |        93%       |       93%     |       93%       |        7s       |
|          25%        |        93%       |       93%     |       93%       |        7s       |
|           0%        |        77%       |       77%     |       77%       |        7s       |

This means that training characteristics and model stability are high.This model has a good generalization,but in terms of accuracy, I think there is still room for improvement by improving the features

图表展示 Performance
=============================
Train 
![image](https://github.com/ArnoldGaius/NLPCC2017_Task2_News_Headline_Categorization/blob/master/image/Train.png)

Test
![image](https://github.com/ArnoldGaius/NLPCC2017_Task2_News_Headline_Categorization/blob/master/image/Test.png)

at this task alpha of MultinomialNB should be 0.1 around 
![image](https://github.com/ArnoldGaius/NLPCC2017_Task2_News_Headline_Categorization/blob/master/image/alpha_num.png)

更多的结果表示方法我已经放在[Text_Classifier用于文本预测分类](https://github.com/ArnoldGaius/Text_Classifier)中，可以直接pip安装使用。<br>
You can get other represent of prediction at [Text_Classifier](https://github.com/ArnoldGaius/Text_Classifier)


