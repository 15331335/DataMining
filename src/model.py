#-*-coding:utf-8-*-
import pandas as pd
import numpy as np


train_data = pd.read_csv('../temp/train.csv', low_memory=False)
test_data = pd.read_csv('../temp/test.csv', low_memory=False)

uid = test_data.uid.values

to_drop = [
    # 需要去掉的特征
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

train = train_data.drop(to_drop, axis=1)
test = test_data.drop(to_drop, axis=1)

import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# stacking: 模型融合
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score


# 模型封装和融合
ntrain = train.shape[0]  # 4999
ntest = test.shape[0]  # 2000

print ntrain, ntest

SEED = 0
NFOLDS = 5
kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED, shuffle=True)

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        # params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain, ))
    oof_test = np.zeros((ntest, ))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    for i, (train_index, test_index) in enumerate(kf):
        print i
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        clf.train(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i,:] = clf.predict(x_test)

    
    oof_test[:] = oof_test_skf.mean(axis=0)
    print f1_score(y_train, oof_train)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    # 'n_estimators': 500,
    #  'warm_start': True, 
    #  #'max_features': 0.2,
    # 'max_depth': 6,
    # 'min_samples_leaf': 2,
    # 'max_features' : 'sqrt',
    # 'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    # 'n_estimators':500,
    # #'max_features': 0.5,
    # 'max_depth': 8,
    # 'min_samples_leaf': 2,
    # 'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'C' : 10,
    'gamma': 1
}

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# test
kn_params = {}
dt_params = {}
gnb_params = {}
lda_params = {}
qda_params = {}
lr_params = {}

kn = SklearnHelper(clf=KNeighborsClassifier, seed=SEED, params=kn_params)
dt = SklearnHelper(clf=DecisionTreeClassifier, seed=SEED, params=dt_params)
gnb = SklearnHelper(clf=GaussianNB, seed=SEED, params=gnb_params)
lda = SklearnHelper(clf=LinearDiscriminantAnalysis, seed=SEED, params=lda_params)
qda = SklearnHelper(clf=QuadraticDiscriminantAnalysis, seed=SEED, params=qda_params)
lr = SklearnHelper(clf=LogisticRegression, seed=SEED, params=lr_params)



y_train = train['label'].ravel()
train = train.drop(['uid', 'label'], axis=1)
x_train = train.values

uid = test['uid']
test = test.drop(['uid', 'label'], axis=1)
x_test = test.values

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
print("Training is complete")

rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
print("Training is complete")

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost
print("Training is complete")

gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
print("Training is complete")

# svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier
# print("Training is complete")

# test
kn_oof_train, kn_oof_test = get_oof(kn,x_train, y_train, x_test) 
print("kn is complete")
dt_oof_train, dt_oof_test = get_oof(dt,x_train, y_train, x_test) 
print("dt is complete")
gnb_oof_train, gnb_oof_test = get_oof(gnb,x_train, y_train, x_test) 
print("gnb is complete")
lda_oof_train, lda_oof_test = get_oof(lda,x_train, y_train, x_test)
print("lda is complete")
qda_oof_train, qda_oof_test = get_oof(qda,x_train, y_train, x_test)
print("qda is complete")
lr_oof_train, lr_oof_test = get_oof(lr,x_train, y_train, x_test)
print("lr is complete")


x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, dt_oof_train, qda_oof_train), axis=1)
x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, dt_oof_test, qda_oof_test), axis=1)

gbm = xgb.XGBClassifier(
    n_estimators = 2000,
    max_depth = 4,
    min_child_weight = 2,
    gamma = 0.9,                        
    subsample = 0.8,
    colsample_bytree = 0.8,
    objective = 'binary:logistic',
    nthread = -1,
    scale_pos_weight = 1
).fit(x_train, y_train)
predictions = gbm.predict(x_test)
probability = gbm.predict_proba(x_test)

StackingSubmission = pd.DataFrame({'uid': uid, 'label': predictions, 'proba': probability[:,1]})
StackingSubmission.to_csv("../out/StackingSubmission.csv", index=False)



# vote

# rf.train(x_train, y_train)
# rf_out = rf.predict(x_test)
# rf_prob = rf.predict_proba(x_test)

# print("Training is complete")

# ada.train(x_train, y_train)
# ada_out = ada.predict(x_test)
# ada_prob = ada.predict_proba(x_test)

# print("Training is complete")

# gb.train(x_train, y_train)
# gb_out = gb.predict(x_test)
# gb_prob = gb.predict_proba(x_test)

# print("Training is complete")

# svc.train(x_train, y_train)
# svc_out = svc.predict(x_test)
# svc_prob = svc.predict_proba(x_test)

# print("Training is complete")

# predictions = (rf_prob[:, 1] + ada_prob[:, 1] + gb_prob[:, 1] + svc_prob[:, 1]) / 4
# print predictions

# for i in range(len(predictions)):
#     if predictions[i] > 0.5:
#         predictions[i] = 1
#     else:
#         predictions[i] = 0


# StackingSubmission = pd.DataFrame({
#     'uid': uid,
#     'label': predictions,
#     'rf_prob': rf_prob[:,1],
#     'ada_prob': ada_prob[:,1],
#     'gb_prob': gb_prob[:,1],
#     'svc_prob': svc_prob[:,1]
# })
# StackingSubmission['prob'] = (StackingSubmission.rf_prob + StackingSubmission.ada_prob + StackingSubmission.gb_prob + StackingSubmission.svc_prob)/4
# StackingSubmission.to_csv("../out/StackingSubmission.csv", index=False)
