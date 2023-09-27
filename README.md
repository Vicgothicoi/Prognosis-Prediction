基于机器学习的临终患者生存期预测，我和两个组员完成了该项目。
首先运行"preprocess.py"生成预处理后的".xlsx"数据，然后直接运行"prognosisPrediction.ipynb"可以实现回归预测和分类预测。
"featureSelection.py"定义了可能使用到的六种特征选择方法，直接运行该模块可以比较这六种方法。
"classifyModel.py"定义了两种分类模型。
"regModel.py"定义了两种回归模型，直接运行该模型以找到最佳参数。
本项目主要使用到的优化方案为SMOTE过采样，特征选择，将特征值转换为遵循正态分布，GridSearchCV模型自动调参。
