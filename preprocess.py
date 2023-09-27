#运行此程序以生成预处理后数据
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

#筛选
data=pd.read_excel("Hospice data-4.14.xlsx")
trimedData=data.loc[(data.loc[:,'转归原始']=='死亡')]
trimedData=trimedData.drop(['患者编号','职业原始','学历原始','宗教原始','转介途径','入院主要诊断','入院诊断分类','病情知晓度原始','入院时神志原始',\
                      '血压原始','入院方式原始','支付方式原始','转归原始','转归'],axis=1)     #剔除文字类的已分类数据
#trimedData=trimedData.drop(['年龄','入院意识状态原始分类','呼吸原始','脉搏原始','体温原始','收缩压','舒张压'],axis=1)    #剔除已分类，但非文字类，原始数据可能有用的数据
#trimedData=trimedData.drop(['机构号','性别','年龄分组','职业','学历','婚姻状况','宗教','民族','转介途径分类',\
#                            '病情知晓度','入院方式','支付方式'],axis=1)    #手动剔除一些数据
#剔除后两项会导致模型性能下降！

trimedData=trimedData.reset_index(drop=True)

#填充
rows, columns = trimedData.shape

# 遍历数据集每一种特征
for i in range(columns):
    # 利用该特征的所有非NaN特征求取均值
    meanVal = round(np.nanmean(trimedData.iloc[:, i]))
    # 将该特征中所有NaN特征全部用均值替换
    trimedData.iloc[ np.isnan(trimedData.iloc[:, i]),i] = meanVal

#SMOTE过采样.   Y:是否肿瘤
smote = SMOTE(sampling_strategy="auto")
X=trimedData.iloc[:,1:]
Y=trimedData.iloc[:,0]
X_resampled, y_resampled = smote.fit_resample(X, Y)
X_resampled = X_resampled.round()
y_resampled = y_resampled.round()
trimedData=pd.concat([y_resampled,X_resampled],axis=1)
#SMOTE过采样效果比较好

trimedData.to_excel('Trimed data.xlsx',index=False) #不保留行索引，header=None则不保留列索引