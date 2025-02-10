import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

datapath = 'D:\\Python\\code\\python_Train\\input\\processed_train.csv'

# 1.加载数据
data = pd.read_csv(datapath)

# 2.处理数据缺失值
# 处理Age缺失值，使用中位数填充
data['Age'].fillna(data['Age'].median(), inplace=True)

# 处理Embarked缺失值，使用众数填充
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# 删除Cabin列
data.drop(columns=['Cabin'], inplace=True)

#  Sex 列映射为0/1
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})

# 3.简化Title特征：将不常见的头衔合并为'Other'
title_mapping = {
    'Col': 'Other', 'Countess': 'Other', 'Don': 'Other', 'Dr': 'Other',
    'Jonkheer': 'Other', 'Lady': 'Other', 'Major': 'Other', 'Master': 'Master',
    'Miss': 'Miss', 'Mlle': 'Miss', 'Mme': 'Mrs', 'Mr': 'Mr', 'Mrs': 'Mrs',
    'Ms': 'Mrs', 'Rev': 'Other', 'Sir': 'Other'
}
data['Title'] = data['Title'].map(title_mapping)

# 对 Title 列进行独热编码
data = pd.get_dummies(data, columns=['Title'], drop_first=True)

# 4.对 Age 进行分组，创建 Age_group 特征
bins = [0, 12, 18, 40, 60, 100]
labels = ['Child', 'Teenager', 'Adult', 'Middle-Aged', 'Senior']
data['Age_group'] = pd.cut(data['Age'], bins=bins, labels=labels)

# 对 Age_group 列进行独热编码
data = pd.get_dummies(data, columns=['Age_group'], drop_first=True)

# 5.对Embarked列进行独热编码
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)


# 6.选择特征列和目标列
features = ['Pclass', 'Sex', 'Age', 'Fare_scaled', 'Family_Size', 'Title_Mr', 'Title_Mrs', 'Title_Miss', 'Embarked_S', 'Embarked_Q']

X = data[features]
y = data['Survived']

# 7.划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义要搜索的超参数范围，使用随机分布来选择参数
param_dist = {
    'n_estimators': randint(50, 200),  # 随机选择树的数量
    'max_depth': [None, 10, 20, 30, 40, 50],  # 树的最大深度
    'min_samples_split': randint(2, 10),  # 分裂一个节点所需的最小样本数
    'min_samples_leaf': randint(1, 4)     # 叶子节点的最小样本数
}

# 8.初始化并训练随机森林模型
rf_model = RandomForestClassifier(random_state=42)

# 使用 RandomizedSearchCV 进行超参数调优，n_iter=100 表示随机选择 100 次参数组合
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=42)

# 训练模型
random_search.fit(X_train, y_train)

# 输出最佳参数和最好的交叉验证得分
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# 训练好的最佳模型
best_rf_model_random = random_search.best_estimator_

rf_model_on_full_data = RandomForestClassifier(random_state=1,
                                               n_estimators=random_search.best_params_['n_estimators'],
                                               max_depth=random_search.best_params_['max_depth'],
                                               min_samples_split=random_search.best_params_['min_samples_split'],
                                               min_samples_leaf=random_search.best_params_['min_samples_leaf'])

rf_model_on_full_data.fit(X, y)

# 读取测试数据
test_data_path = '../input/processed-data/processed_test.csv'
test_data = pd.read_csv(test_data_path)

# 填充缺失值（与训练数据一致）
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)  # 使用训练集的中位数

# 处理Embarked缺失值，使用众数填充
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

# 删除Cabin列
test_data.drop(columns=['Cabin'], inplace=True)

#  Sex 列映射为0/1
test_data['Sex'] = test_data['Sex'].map({'female': 0, 'male': 1})

# 3.简化Title特征：将不常见的头衔合并为'Other'
title_mapping = {
    'Col': 'Other', 'Countess': 'Other', 'Don': 'Other', 'Dr': 'Other',
    'Jonkheer': 'Other', 'Lady': 'Other', 'Major': 'Other', 'Master': 'Master',
    'Miss': 'Miss', 'Mlle': 'Miss', 'Mme': 'Mrs', 'Mr': 'Mr', 'Mrs': 'Mrs',
    'Ms': 'Mrs', 'Rev': 'Other', 'Sir': 'Other'
}
test_data['Title'] = test_data['Title'].map(title_mapping)

# 对 Title 列进行独热编码
test_data = pd.get_dummies(test_data, columns=['Title'], drop_first=True)

# 4.对 Age 进行分组，创建 Age_group 特征
bins = [0, 12, 18, 40, 60, 100]
labels = ['Child', 'Teenager', 'Adult', 'Middle-Aged', 'Senior']
test_data['Age_group'] = pd.cut(test_data['Age'], bins=bins, labels=labels)

# 对 Age_group 列进行独热编码
test_data = pd.get_dummies(test_data, columns=['Age_group'], drop_first=True)

# 5.对Embarked列进行独热编码
test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)

# 使用与训练时相同的特征列
test_X = test_data[features]

# 使用训练好的模型进行预测
test_preds = rf_model_on_full_data.predict(test_X)

# 生成提交文件
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_preds})
output.to_csv('submission.csv', index=False)

