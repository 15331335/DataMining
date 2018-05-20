#-*-coding:utf-8-*-
import pandas as pd

# 声明数据列的名称
uid_train_names = [
    'uid',
    'label'
]
voice_train_names = [
    'uid',
    'opp_num',  # 通话对方的号码（加密）
    'opp_head',
    'opp_len',
    'start_time',
    'end_time',
    'call_type',
    'in_out'
]
sms_train_names = [
    'uid',
    'opp_num',
    'opp_head',
    'opp_len',
    'start_time',
    'in_out'
]
wa_train_names = [  # 网页和应用的访问记录
    'uid',
    'wa_name',
    'visit_cnt',  # 一天中的访问次数
    'visit_dura',
    'up_flow',
    'down_flow',
    'wa_type',
    'date'
]

# 载入数据
uid_train = pd.read_table('../data/train/uid_train.txt', header=None, names=uid_train_names, low_memory=False)
voice_train = pd.read_table('../data/train/voice_train.txt', header=None, names=voice_train_names, low_memory=False)
sms_train = pd.read_table('../data/train/sms_train.txt', header=None, names=sms_train_names, low_memory=False)
wa_train = pd.read_table('../data/train/wa_train.txt', header=None, names=wa_train_names, low_memory=False)

# 导出数据
uid_train.to_csv("../temp/uid_train.csv", index=False)
voice_train.to_csv("../temp/voice_train.csv", index=False)
sms_train.to_csv("../temp/sms_train.csv", index=False)
wa_train.to_csv("../temp/wa_train.csv", index=False)


# 导出测试 A(uid=u5000~u6999)
uid_test_a = uid_train.loc[:1999, :].copy(deep=True)  # 2000
uid = 5000
for i in range(2000):
    uid_test_a.at[i, 'uid'] = 'u'+str(uid)
    uid_test_a.at[i, 'label'] = -1
    uid = uid+1
uid_test_a.to_csv("../temp/uid_test_a.csv", index=False)

voice_test_a = pd.read_table('../data/testA/voice_test_a.txt', header=None, names=voice_train_names, low_memory=False)
sms_test_a = pd.read_table('../data/testA/sms_test_a.txt', header=None, names=sms_train_names, low_memory=False)
wa_test_a = pd.read_table('../data/testA/wa_test_a.txt', header=None, names=wa_train_names, low_memory=False)

voice_test_a.to_csv("../temp/voice_test_a.csv", index=False)
sms_test_a.to_csv("../temp/sms_test_a.csv", index=False)
wa_test_a.to_csv("../temp/wa_test_a.csv", index=False)


