#运行此代码以查看不同特征选择间的效果比较
from sklearn.linear_model import (LinearRegression, Ridge,Lasso)  
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler  
from sklearn.ensemble import RandomForestRegressor    
from minepy import MINE  
import numpy as np
import pandas as pd


def MIC(X,Y):
    #使用最大信息系数进行特征选择  
    m = MINE()
    MIC=[]
    for i in range(len(X[0])):   
        m.compute_score(X[:,i],Y)  
        MIC.append(m.mic())

    sortedMIC=sorted(MIC,reverse=True)#默认升序，变为降序
    symbol=[]
    for i in range(10):#选出MIC值前十的索引
        symbol.append(MIC.index(sortedMIC[i]))
    newX=X[:,symbol]

    return symbol

def Lr_reg(X,Y):
    #使用线性模型的回归系数进行特征选择
    lr = LinearRegression()  
    lr.fit(X, Y)

    coefficient=list(np.abs(lr.coef_))#回归系数
    sortedLR=sorted(coefficient,reverse=True)
    symbol=[]
    for i in range(10):#选出回归系数前十的索引
        symbol.append(coefficient.index(sortedLR[i]))
    newX=X[:,symbol]

    return symbol

def ridge(X,Y):
    #使用Ridge回归的回归系数进行特征选择
    ridge = Ridge(alpha=7)  
    ridge.fit(X, Y)  
        
    coefficient=list(np.abs(ridge.coef_))#回归系数
    sortedRidge=sorted(coefficient,reverse=True)
    symbol=[]
    for i in range(10):#选出回归系数前十的索引
        symbol.append(coefficient.index(sortedRidge[i]))
    newX=X[:,symbol]

    return symbol

def lasso(X,Y):
    #使用Lasso回归的回归系数进行特征选择，不过效果太差
    lasso = Lasso(alpha=7)  
    lasso.fit(X, Y)  
        
    coefficient=list(np.abs(lasso.coef_))#回归系数
    sortedLasso=sorted(coefficient,reverse=True)
    symbol=[]
    for i in range(10):#选出回归系数前十的索引
        symbol.append(coefficient.index(sortedLasso[i]))
    newX=X[:,symbol]

    return symbol

def rfe(X,Y):
    #使用递归特征消除进行特征选择
    lr = LinearRegression()  
    lr.fit(X, Y)  
    rfe = RFE(lr, n_features_to_select=5)  
    rfe.fit(X,Y)  

    symbol=list(rfe.ranking_[0:10])
    newX=X[:,symbol]

    return symbol

def rf(X,Y):
    #使用随机森林回归器的特征重要性进行特征选择
    rf = RandomForestRegressor()  
    rf.fit(X,Y)  

    imp=list(rf.feature_importances_)#特征重要性
    sortedImp=sorted(imp,reverse=True)
    symbol=[]
    for i in range(10):#选出回归系数前十的索引
        symbol.append(imp.index(sortedImp[i]))
    newX=X[:,symbol]
    
    return symbol

def main(X,Y):
    names = ["x%s" % i for i in range(1,31)]  
  
    ranks = {}  
    
    def rank_to_dict(ranks, names, order=1):  
        minmax = MinMaxScaler()  
        ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]  
        ranks = map(lambda x: round(x, 2), ranks)  
        return dict(zip(names, ranks ))  
    
    lr = LinearRegression()  
    lr.fit(X, Y)  
    ranks["Lr reg"] = rank_to_dict(np.abs(lr.coef_), names)  
    
    ridge = Ridge(alpha=7)  
    ridge.fit(X, Y)  
    ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)  
    
    lasso = Lasso(alpha=.05)  
    lasso.fit(X, Y)  
    ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)   
    
    #stop the search when 5 features are left (they will get equal scores)  
    rfe = RFE(lr, n_features_to_select=5)  
    rfe.fit(X,Y)  
    ranks["RFE"] = rank_to_dict(list(map(float, rfe.ranking_)), names, order=-1)  
    
    rf = RandomForestRegressor()  
    rf.fit(X,Y)  
    ranks["RF"] = rank_to_dict(rf.feature_importances_, names)    
    
    mine = MINE()  
    mic_scores = []  
    for i in range(X.shape[1]):  
        mine.compute_score(X[:,i], Y)  
        m = mine.mic()  
        mic_scores.append(m)    
    ranks["MIC"] = rank_to_dict(mic_scores, names)    
    
    r = {}  
    for name in names:  
        r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)    
    methods = sorted(ranks.keys())  
    ranks["Mean"] = r  
    methods.append("Mean")  
    
    # ranks = pd.DataFrame(ranks)
    print("\t%s" % "\t".join(methods))  
    for name in names:  
        print("%s\t%s" % (name, "\t".join(map(str,[ranks[method][name] for method in methods]))))

if __name__ == '__main__':
    data=pd.read_excel("Trimed data.xlsx")
    Y=data.values[:,28]
    X=np.delete(data.values,28,axis=1)
    main(X,Y)