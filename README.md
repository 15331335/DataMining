# DataMining

JDATA 基于移动网络通讯行为的风险用户识别



## Feature List

以下四类特征均为对单个用户而言：

- **cnt** 为基础的数量特征。
- **unique** 为涉及维度的变量集合的元素数量。
- **onehot** 为涉及维度的不同类型的数量特征。
- **val** 为涉及维度的统计特征，包括标准差、最大值、最小值、中位数、均值、总和。

| 变量                   | 类型   | 维度             | 描述                         |
| ---------------------- | ------ | ---------------- | ---------------------------- |
| voice_cnt              | cnt    | 通话记录         | 采样期间内的通话记录数量     |
| voice_unique_num_cnt   | unique | 通话记录         | 涉及的号码数量               |
| voice_onehot_len_cnt   | onehot | 通话记录         | 各种对端号码长度的数量       |
| voice_onehot_type_cnt  | onehot | 通话记录         | 各种通话类型的数量           |
| voice_onehot_inout_cnt | onehot | 通话记录         | 不同主被叫类型的数量         |
| voice_dura_val         | val    | 通话记录         | 通话时长的统计特征           |
| voice_onehot_head_cnt  | onehot | 通话记录         | 各种对端号码头的数量         |
| sms_cnt                | cnt    | 短信记录         | 采样期间内的短信记录数量     |
| sms_unique_num_cnt     | unique | 短信记录         | 涉及的号码数量               |
| sms_onehot_len_cnt     | onehot | 短信记录         | 各种对端号码长度的数量       |
| sms_onehot_inout_cnt   | onehot | 短信记录         | 发送/接收类型的数量          |
| sms_onehot_head_cnt    | onehot | 短信记录         | 各种对端号码头的数量         |
| sms_onehot_hour_cnt    | onehot | 短信记录         | 不同时间段（24h）内的数量    |
| wa_unique_name_cnt     | unique | 网站/App访问记录 | 涉及的网站/App数量           |
| wa_onehot_type_cnt     | onehot | 网站/App访问记录 | 网站/App的数量               |
| wa_up_flow_val         | val    | 网站/App访问记录 | 上行流量的统计特征           |
| wa_down_flow_val       | val    | 网站/App访问记录 | 下行流量的统计特征           |
| wa_special_dura_cnt    | cnt    | 网站/App访问记录 | 访问时长为零的数量           |
| wa_up_speed_val        | val    | 网站/App访问记录 | 关联特征，上行速度的统计特征 |
| wa_down_speed_val      | val    | 网站/App访问记录 | 关联特征，下行速度的统计特征 |
| wa_dura_per_visit_val  | val    | 网站/App访问记录 | 关联特征，单次时长的统计特征 |





