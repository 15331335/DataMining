#-*-coding:utf-8-*-
import pandas as pd
import numpy as np


train_data = pd.read_csv('../temp/train.csv', low_memory=False)

to_drop = [
    # 需要去掉的特征
    'uid',
    'voice_out_cnt',
    'voice_in_cnt',
    'voice_out_rate',
    'voice_type1_cnt',
    'voice_type2_cnt',
    'voice_type3_cnt',
    'voice_type4_cnt',
    'voice_type5_cnt',
    'voice_total_time',
    'voice_avg_time',
    'voice_len11_cnt',
    'voice_opp_cnt',
    'voice_out_opp_cnt',
    'voice_in_opp_cnt',

    'sms_out_cnt',
    'sms_in_cnt',
    'sms_out_rate',
    'sms_opp_cnt',
    'sms_out_opp_cnt',
    'sms_in_opp_cnt',
    'sms_len11_cnt',
    'sms_106_cnt',

    'wa_web_cnt',
    'wa_app_cnt',
    'wa_web_rate',
    'wa_opp_cnt',
    'wa_web_opp_cnt',
    'wa_app_opp_cnt',
    'wa_app_opp_rate'
]

test = train_data.drop(to_drop, axis=1).values


import matplotlib.pyplot as plt
import seaborn as sns

# 模型测试
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability = True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()
]

log_cols = ['Classifier', 'Accuracy']
log = pd.DataFrame(columns = log_cols)

sss = StratifiedShuffleSplit(n_splits = 10, test_size = 0.1, random_state = 0)

X = test[0::, 1::]  # 特征列（不包含 uid）
y = test[0::, 0]  # 标签列

acc_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc
        
for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0  # mean for 10 splits
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns = log_cols)
    log = log.append(log_entry)
    
plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

print log

sns.set_color_codes('muted')
sns.barplot(x = 'Accuracy', y = 'Classifier', data = log, color = 'b')
plt.show()