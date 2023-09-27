from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn import model_selection
import numpy as np
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def XGboostClassifier(newX, Y):   
    # 将标签转换为多分类任务的形式
    def classify_label(y):
        if y <= 4:
            return 0
        elif y <= 9:
            return 1
        elif y <= 23.8:
            return 2
        elif y <= 73:
            return 3
        else:
            return 4

    classify_label = np.vectorize(classify_label)

    Y = classify_label(Y)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(newX, Y, test_size=0.3)

    model = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=100)
    # 创建XGBoost多分类模型，并指定一些参数，如最大树深度（max_depth）、学习率（learning_rate）、估计器数量（n_estimators）和随机种子（random_state）
    model.fit(X_train, Y_train)
    test_predict = model.predict(X_test)
    # 对测试数据集（X_test）进行预测，返回预测结果

    # 创建一个SHAP解释器
    explainer = shap.Explainer(model)

    # 计算SHAP值
    shap_values = explainer.shap_values(X_test)

    # 打印SHAP值
    shap.summary_plot(shap_values, X_test, plot_type='bar')

    # 计算混淆矩阵
    confusion_mat = confusion_matrix(Y_test, test_predict)

    # 打印混淆矩阵
    print("混淆矩阵:")
    print(confusion_mat)

    # 计算准确性（Accuracy）
    accuracy = accuracy_score(Y_test, test_predict)

    # 计算精确率（Precision）
    precision = precision_score(Y_test, test_predict, average='macro')

    # 计算召回率（Recall）
    recall = recall_score(Y_test, test_predict, average='macro')

    # 计算F1值
    f1 = f1_score(Y_test, test_predict, average='macro')

    # 打印结果
    print("准确性 (Accuracy):", accuracy)
    print("精确率 (Precision):", precision)
    print("召回率 (Recall):", recall)
    print("F1 值 (F1 Score):", f1)
    print("测试集真实标签：", Y_test)
    print("预测标签：", test_predict)

def GBDTClassifier(newX, Y):
    def classify_label(y):
        if y <= 4:
            return 0
        elif y <= 9:
            return 1
        elif y <= 23.8:
            return 2
        elif y <= 73:
            return 3
        else:
            return 4

    classify_label = np.vectorize(classify_label)

    Y = classify_label(Y)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(newX, Y, test_size=0.3)

    model = GradientBoostingClassifier(max_depth=6, learning_rate=0.05, n_estimators=100)
    model.fit(X_train, Y_train)
    test_predict = model.predict(X_test)

    # 计算混淆矩阵
    confusion_mat = confusion_matrix(Y_test, test_predict)

    # 打印混淆矩阵
    print("混淆矩阵:")
    print(confusion_mat)

    # 计算准确性（Accuracy）
    accuracy = accuracy_score(Y_test, test_predict)

    # 计算精确率（Precision）
    precision = precision_score(Y_test, test_predict, average='macro')

    # 计算召回率（Recall）
    recall = recall_score(Y_test, test_predict, average='macro')

    # 计算F1值
    f1 = f1_score(Y_test, test_predict, average='macro')

    # 打印结果
    print("准确性 (Accuracy):", accuracy)
    print("精确率 (Precision):", precision)
    print("召回率 (Recall):", recall)
    print("F1 值 (F1 Score):", f1)
    print("测试集真实标签：", Y_test)
    print("预测标签：", test_predict)