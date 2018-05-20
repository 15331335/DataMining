#-*-coding:utf-8-*-
import pandas as pd
import numpy as np


data = pd.read_csv('../out/StackingSubmission.csv', low_memory=False)
sorted_data = data.sort_values('proba', ascending=False)
temp = sorted_data[['uid', 'label']]
temp.to_csv("../out/submit1.csv", index=False)

