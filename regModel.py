#运行此程序以得到模型最佳参数
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, KFold     #GridSearchCV: 网格搜索和交叉验证, 模型自动调参
from sklearn.compose import TransformedTargetRegressor    #对转换后的目标进行回归
from sklearn.preprocessing import QuantileTransformer   #将特征转换为遵循均匀或正态分布，以减少离群值影响
import xgboost as xgb
import numpy as np
import pandas as pd
from featureSelection import *

def XGboost(newX,Y):
    model = xgb.XGBRegressor(learning_rate=0.3,max_depth=7,n_estimators=80)
    # 将特征值转换为遵循正态分布以减少离群值影响，n_quantiles：要计算的分位数的数量
    regr_trans = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(n_quantiles=200,output_distribution='normal'))
        
    # 在训练集上训练模型
    X_train,X_test,Y_train,Y_test=train_test_split(newX,Y,test_size=0.3)

    regr_trans.fit(X_train, Y_train)
    # 在测试集上进行预测
    test_predict=regr_trans.predict(X_test)
        
    # 计算误差和R2 score
    mape = mean_absolute_percentage_error(Y_test, test_predict)
    r2 = r2_score(Y_test, test_predict)

    print("绝对值百分比误差：", mape)
    print("R2 score: ", r2)


def GBDT(newX,Y):
    # 创建GBDT模型
    model = GradientBoostingRegressor(learning_rate=0.05, max_depth=6, n_estimators=100)
    regr_trans = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(n_quantiles=200,output_distribution='normal'))

    # 使用四折交叉验证
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    # 定义用于保存每次交叉验证结果的列表
    mape_scores = []
    r2_scores = []

    # 进行交叉验证并记录测试集生存期和预测生存期
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(newX)):
        X_train, X_test = newX[train_idx], newX[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        # 在训练集上训练模型
        regr_trans.fit(X_train, Y_train)
        
        # 在测试集上进行预测
        test_predict = regr_trans.predict(X_test)
        
        # 计算误差和R2 score
        mape = mean_absolute_percentage_error(Y_test, test_predict)
        r2 = r2_score(Y_test, test_predict)
        
        mape_scores.append(mape)
        r2_scores.append(r2)

        # 打印结果
        print(f"第{fold_idx+1}折的平均绝对值百分比误差：", mape)
        print(f"第{fold_idx+1}折的R2 score: ", r2)

    mean_mape = np.mean(mape_scores)
    std_mape = np.std(mape_scores)
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    print("绝对值百分比误差的平均值：", mean_mape)
    print("绝对值百分比误差的标准差：", std_mape)
    print("R2 score的平均值: ", mean_r2)
    print("R2 score的标准差: ", std_r2)

def main(newX, Y):
    X_train,X_test,y_train,y_test=train_test_split(newX,Y,test_size=0.3)

    param_grid = {'learning_rate' :[0.05, 0.1, 0.3],    #学习率
                'max_depth': [6, 7, 8],    #每一棵树最大深度
                'n_estimators': [100, 90, 80, 70, 60],      #迭代次数
                }
    
    #refit=True 以交叉验证训练集得到最佳参数, verbose = 1 偶尔输出训练过程                
    grid = GridSearchCV(xgb.XGBRegressor(), param_grid, refit = True, verbose = 1, n_jobs=-1)      
    regr_trans = TransformedTargetRegressor(regressor=grid, transformer=QuantileTransformer(n_quantiles=200,output_distribution='normal'))

    # fitting the model for grid search 
    grid_result=regr_trans.fit(newX,Y)
    best_params=grid_result.regressor_.best_params_
    print(best_params)

    #using best params to create and fit model
    best_model = xgb.XGBRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"], learning_rate=best_params["learning_rate"])
    regr_trans = TransformedTargetRegressor(regressor=best_model, transformer=QuantileTransformer(n_quantiles=200,output_distribution='normal'))
    regr_trans.fit(X_train, y_train)
    yhat = regr_trans.predict(X_test)

    #evaluate metrics
    print(mean_absolute_percentage_error(y_test, yhat))
    print(r2_score(y_test, yhat))

if __name__ == '__main__':
    data=pd.read_excel("Trimed data.xlsx")
    Y=data.values[:,-5]
    Y[Y == 0] = 1 #对生存期等于0的患者进行修改，否则会导致计算误差时数值溢出
    X=np.delete(data.values,-5,axis=1)
    symbol1=MIC(X,Y)
    newX=X[:,symbol1[0:5]]
    main(newX,Y)