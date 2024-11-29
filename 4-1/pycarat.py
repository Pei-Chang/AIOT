# Install PyCaret if not installed
# pip install pycaret

import pandas as pd
from pycaret.classification import setup, compare_models, tune_model, finalize_model, predict_model

# 加载数据
train_data = pd.read_csv('./AIOT/Pycarat/train.csv')
test_data = pd.read_csv('./AIOT/Pycarat/test.csv')

# 检查数据
print(train_data.head())

# 数据预处理
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

# 从 Name 中提取 Title
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# 统一 Title 分组
title_map = {
    'Miss': 'Miss', 'Mlle': 'Miss', 'Ms': 'Miss',
    'Mr': 'Mr',
    'Mrs': 'Mrs', 'Mme': 'Mrs',
    'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 
    # 其他 Title 映射...
}
train_data['Title'] = train_data['Title'].map(title_map).fillna('Rare')
test_data['Title'] = test_data['Title'].map(title_map).fillna('Rare')

# 删除无关特征
train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 填补缺失值
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)

train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

# One-hot 编码
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

# 确保测试集和训练集的列对齐
test_data = test_data.reindex(columns=train_data.columns.drop('Survived'), fill_value=0)

# PyCaret 设置
clf = setup(data=train_data, target='Survived', session_id=42, preprocess=True, verbose=False)

# 模型比较
best_model = compare_models()

# 调参
tuned_model = tune_model(best_model)

# 模型最终化
final_model = finalize_model(tuned_model)

# 对测试集进行预测
predictions = predict_model(final_model, data=test_data)

# 检查返回结果的列名
print(predictions.columns)

# 提取预测结果
if 'Label' in predictions.columns:
    test_data['Survived'] = predictions['Label']
elif 'prediction_label' in predictions.columns:
    test_data['Survived'] = predictions['prediction_label']
else:
    raise ValueError("No valid prediction column found in predictions DataFrame.")
# 保存预测结果
submission = pd.DataFrame({
    'PassengerId': pd.read_csv('./AIOT/Pycarat/test.csv')['PassengerId'],
    'Survived': test_data['Survived']
})
submission.to_csv('./AIOT/Pycarat/submission.csv', index=False)

print("预测结果已保存为 submission.csv")