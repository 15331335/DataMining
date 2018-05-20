#-*-coding:utf-8-*-
import pandas as pd
import numpy as np


# voice feature
def get_voice_feature(base, dataset):
    # 加入特征：每个用户的通话数量（采样期间内）
    grouped_temp = dataset.groupby('uid')
    counted_temp = pd.DataFrame({'voice_cnt': grouped_temp['in_out'].count()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')

    # 加入特征：每个用户打出的通话数量
    grouped_temp = dataset[dataset.in_out==0].groupby('uid')
    counted_temp = pd.DataFrame({'voice_out_cnt': grouped_temp['in_out'].count()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 加入特征：每个用户打入的通话数量
    grouped_temp = dataset[dataset.in_out==1].groupby('uid')
    counted_temp = pd.DataFrame({'voice_in_cnt': grouped_temp['in_out'].count()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 加入特征：每个用户打出/打入的通话记录比例
    base['voice_out_rate'] = base.voice_out_cnt / base.voice_cnt
    base['voice_in_rate'] = base.voice_in_cnt / base.voice_cnt

    # 加入特征：各个通话类型的数量
    # 类型 1：本地
    grouped_temp = dataset[dataset.call_type==1].groupby('uid')
    counted_temp = pd.DataFrame({'voice_type1_cnt': grouped_temp['call_type'].count()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 类型 2：省内长途
    grouped_temp = dataset[dataset.call_type==2].groupby('uid')
    counted_temp = pd.DataFrame({'voice_type2_cnt': grouped_temp['call_type'].count()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 类型 3：省际长途
    grouped_temp = dataset[dataset.call_type==3].groupby('uid')
    counted_temp = pd.DataFrame({'voice_type3_cnt': grouped_temp['call_type'].count()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 类型 4：港澳台长途
    grouped_temp = dataset[dataset.call_type==4].groupby('uid')
    counted_temp = pd.DataFrame({'voice_type4_cnt': grouped_temp['call_type'].count()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 类型 5：国际长途
    grouped_temp = dataset[dataset.call_type==5].groupby('uid')
    counted_temp = pd.DataFrame({'voice_type5_cnt': grouped_temp['call_type'].count()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 加入特征：各个通话类型的比例
    base['voice_type1_rate'] = base.voice_type1_cnt / base.voice_cnt
    base['voice_type2_rate'] = base.voice_type2_cnt / base.voice_cnt
    base['voice_type3_rate'] = base.voice_type3_cnt / base.voice_cnt
    base['voice_type4_rate'] = base.voice_type4_cnt / base.voice_cnt
    base['voice_type5_rate'] = base.voice_type5_cnt / base.voice_cnt

    # 提取并记录通话开始时间的各个分量（天、时、分、秒）
    temp_day = dataset.start_time / 1000000
    dataset['start_dd'] = temp_day.astype(int)
    temp_hour = dataset.start_time % 1000000 / 10000
    dataset['start_hh'] = temp_hour.astype(int)
    temp_minute = dataset.start_time % 10000 / 100
    dataset['start_mm'] = temp_minute.astype(int)
    temp_second = dataset.start_time % 100
    dataset['start_ss'] = temp_second.astype(int)
    # 提取并记录通话结束时间的各个分量（天、时、分、秒）
    temp_day = dataset.end_time / 1000000
    dataset['end_dd'] = temp_day.astype(int)
    temp_hour = dataset.end_time % 1000000 / 10000
    dataset['end_hh'] = temp_hour.astype(int)
    temp_minute = dataset.end_time % 10000 / 100
    dataset['end_mm'] = temp_minute.astype(int)
    temp_second = dataset.end_time % 100
    dataset['end_ss'] = temp_second.astype(int)
    # 计算每个通话记录的时长
    temp_start = dataset['start_dd']*24*60*60 + dataset['start_hh']*60*60 + dataset['start_mm']*60 + dataset['start_ss']
    temp_end = dataset['end_dd']*24*60*60 + dataset['end_hh']*60*60 + dataset['end_mm']*60 + dataset['end_ss']
    dataset['call_dura'] = temp_end - temp_start

    # 加入特征：每个用户的通话记录总时长
    grouped_temp = dataset.groupby('uid')
    counted_temp = pd.DataFrame({'voice_total_time': grouped_temp['call_dura'].sum()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 加入特征：每个用户的平均通话时长
    base['voice_avg_time'] = base.voice_total_time / base.voice_cnt
    # 加入特征：每个用户的通话记录时长的标准差
    grouped_temp = dataset.groupby('uid')
    counted_temp = pd.DataFrame({'voice_time_std': grouped_temp['call_dura'].std()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 加入特征：每个用户的通话记录时长的均值
    grouped_temp = dataset.groupby('uid')
    counted_temp = pd.DataFrame({'voice_time_mean': grouped_temp['call_dura'].mean()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')

    # 加入特征：通话开始时间（24h）的平均值
    grouped_temp = dataset.groupby('uid')
    counted_temp = pd.DataFrame({'voice_start_hh_mean': grouped_temp['start_hh'].mean()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 加入特征：通话开始时间（24h）的标准差
    grouped_temp = dataset.groupby('uid')
    counted_temp = pd.DataFrame({'voice_start_hh_std': grouped_temp['start_hh'].std()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')


    # 加入特征：每个用户的通话记录中出现的不同对象数量
    grouped_temp = dataset.groupby('uid')
    counted_temp = pd.DataFrame({'voice_opp_cnt': grouped_temp['opp_num'].unique().apply(lambda x: len(x))}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 加入特征：通话记录中出现的不同对象比例
    base['voice_opp_rate'] = base.voice_opp_cnt / base.voice_cnt

    # 加入特征：打出/打出记录中不同对象数量，以及比例
    grouped_temp = dataset[dataset.in_out==0].groupby('uid')
    counted_temp = pd.DataFrame({'voice_out_opp_cnt': grouped_temp['opp_num'].unique().apply(lambda x: len(x))}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    base['voice_out_opp_rate'] = base.voice_out_opp_cnt / base.voice_out_cnt

    grouped_temp = dataset[dataset.in_out==1].groupby('uid')
    counted_temp = pd.DataFrame({'voice_in_opp_cnt': grouped_temp['opp_num'].unique().apply(lambda x: len(x))}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    base['voice_in_opp_rate'] = base.voice_in_opp_cnt / base.voice_in_cnt

    # 加入特征：通话记录中号码长度为 11 的数量及比例
    grouped_temp = dataset[dataset.opp_len==11].groupby('uid')
    counted_temp = pd.DataFrame({'voice_len11_cnt': grouped_temp['opp_len'].count()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    base['len11_rate'] = base.voice_len11_cnt / base.voice_cnt

    base = base.fillna(0)
    return base

# sms feature
def get_sms_feature(base, dataset):
    # 加入特征：每个用户的通信数量（采样期间内）
    grouped_temp = dataset.groupby('uid')
    counted_temp = pd.DataFrame({'sms_cnt': grouped_temp['in_out'].count()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')

    # 加入特征：每个用户发送的通信数量
    grouped_temp = dataset[dataset.in_out==0].groupby('uid')
    counted_temp = pd.DataFrame({'sms_out_cnt': grouped_temp['in_out'].count()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 加入特征：每个用户接受的通信数量
    grouped_temp = dataset[dataset.in_out==1].groupby('uid')
    counted_temp = pd.DataFrame({'sms_in_cnt': grouped_temp['in_out'].count()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 加入特征：每个用户发送/接受的通信记录比例
    base['sms_out_rate'] = base.sms_out_cnt / base.sms_cnt
    base['sms_in_rate'] = base.sms_in_cnt / base.sms_cnt

    # 提取并记录通信时间的各个分量（天、时、分、秒）
    temp_day = dataset.start_time / 1000000
    dataset['dd'] = temp_day.astype(int)
    temp_hour = dataset.start_time % 1000000 / 10000
    dataset['hh'] = temp_hour.astype(int)
    temp_minute = dataset.start_time % 10000 / 100
    dataset['mm'] = temp_minute.astype(int)
    temp_second = dataset.start_time % 100
    dataset['ss'] = temp_second.astype(int)

    # 加入特征：通信时间（24h）的平均值
    grouped_temp = dataset.groupby('uid')
    counted_temp = pd.DataFrame({'sms_hh_mean': grouped_temp['hh'].mean()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 加入特征：通信时间（24h）的标准差
    grouped_temp = dataset.groupby('uid')
    counted_temp = pd.DataFrame({'sms_hh_std': grouped_temp['hh'].std()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')

    # 加入特征：每个用户的通信记录中出现的不同对象数量
    grouped_temp = dataset.groupby('uid')
    counted_temp = pd.DataFrame({'sms_opp_cnt': grouped_temp['opp_num'].unique().apply(lambda x: len(x))}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 加入特征：通信记录中出现的不同对象比例
    base['sms_opp_rate'] = base.sms_opp_cnt / base.sms_cnt

    # 加入特征：接受/发送记录中不同对象数量，以及比例
    grouped_temp = dataset[dataset.in_out==0].groupby('uid')
    counted_temp = pd.DataFrame({'sms_out_opp_cnt': grouped_temp['opp_num'].unique().apply(lambda x: len(x))}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    base['sms_out_opp_rate'] = base.sms_out_opp_cnt / base.sms_out_cnt

    grouped_temp = dataset[dataset.in_out==1].groupby('uid')
    counted_temp = pd.DataFrame({'sms_in_opp_cnt': grouped_temp['opp_num'].unique().apply(lambda x: len(x))}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    base['sms_in_opp_rate'] = base.sms_in_opp_cnt / base.sms_in_cnt

    # 加入特征：通信记录中号码长度为 11 的数量及比例
    grouped_temp = dataset[dataset.opp_len==11].groupby('uid')
    counted_temp = pd.DataFrame({'sms_len11_cnt': grouped_temp['opp_len'].count()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    base['sms_len11_rate'] = base.sms_len11_cnt / base.sms_cnt

    # 加入特征：通信记录中号码开头为 106 的数量及比例
    grouped_temp = dataset[dataset.opp_head==106].groupby('uid')
    counted_temp = pd.DataFrame({'sms_106_cnt': grouped_temp['opp_head'].count()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    base['sms_106_rate'] = base.sms_106_cnt / base.sms_cnt

    base = base.fillna(0)
    return base

# wa feature
def get_wa_feature(base, dataset):
    # 加入特征：每个用户的访问数量（采样期间内）
    grouped_temp = dataset.groupby('uid')
    counted_temp = pd.DataFrame({'wa_cnt': grouped_temp['visit_cnt'].sum()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')

    # 加入特征：访问记录中网页的数量及比例
    grouped_temp = dataset[dataset.wa_type==0].groupby('uid')
    counted_temp = pd.DataFrame({'wa_web_cnt': grouped_temp['visit_cnt'].sum()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    base['wa_web_rate'] = base.wa_web_cnt / base.wa_cnt
    # 加入特征：访问记录中应用的数量及比例
    grouped_temp = dataset[dataset.wa_type==1].groupby('uid')
    counted_temp = pd.DataFrame({'wa_app_cnt': grouped_temp['visit_cnt'].sum()}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    base['wa_app_rate'] = base.wa_app_cnt / base.wa_cnt

    # 加入特征：访问记录中不同对象的数量
    grouped_temp = dataset.groupby('uid')
    counted_temp = pd.DataFrame({'wa_opp_cnt': grouped_temp['wa_name'].unique().apply(lambda x: len(x))}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    # 加入特征：访问记录中不同网页对象的数量及比例
    grouped_temp = dataset[dataset.wa_type==0].groupby('uid')
    counted_temp = pd.DataFrame({'wa_web_opp_cnt': grouped_temp['wa_name'].unique().apply(lambda x: len(x))}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    base['wa_web_opp_rate'] = base.wa_web_opp_cnt / base.wa_opp_cnt
    # 加入特征：访问记录中不同应用对象的数量及比例
    grouped_temp = dataset[dataset.wa_type==1].groupby('uid')
    counted_temp = pd.DataFrame({'wa_app_opp_cnt': grouped_temp['wa_name'].unique().apply(lambda x: len(x))}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    base['wa_app_opp_rate'] = base.wa_app_opp_cnt / base.wa_opp_cnt

    # 加入特征：访问记录中不同应用对象的数量及比例
    grouped_temp = dataset[dataset.wa_type==1].groupby('uid')
    counted_temp = pd.DataFrame({'wa_app_opp_cnt': grouped_temp['wa_name'].unique().apply(lambda x: len(x))}).reset_index()
    base = pd.merge(base, counted_temp, how='left', on='uid')
    base['wa_app_opp_rate'] = base.wa_app_opp_cnt / base.wa_opp_cnt

    base = base.fillna(0)
    return base


uid_train = pd.read_csv('../temp/uid_train.csv', low_memory=False)
voice_train = pd.read_csv('../temp/voice_train.csv', low_memory=False)
sms_train = pd.read_csv('../temp/sms_train.csv', low_memory=False)
wa_train = pd.read_csv('../temp/wa_train.csv', low_memory=False)

# 导入测试 A
uid_test = pd.read_csv('../temp/uid_test_a.csv', low_memory=False)
voice_test = pd.read_csv('../temp/voice_test_a.csv', low_memory=False)
sms_test = pd.read_csv('../temp/sms_test_a.csv', low_memory=False)
wa_test = pd.read_csv('../temp/wa_test_a.csv', low_memory=False)


uid_train = get_voice_feature(uid_train, voice_train)
uid_train = get_sms_feature(uid_train, sms_train)
uid_train = get_wa_feature(uid_train, wa_train)

uid_test = get_voice_feature(uid_test, voice_test)
uid_test = get_sms_feature(uid_test, sms_test)
uid_test = get_wa_feature(uid_test, wa_test)

# 导出特征
uid_train.to_csv("../temp/train.csv", index=False)
uid_test.to_csv("../temp/test.csv", index=False)

