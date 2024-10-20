@@ -0,0 +1,222 @@
# 利用 CRISP-DM 模板解決波士頓房價預測的多元回歸問題(取出重要特徵)
#### Prompt: 
#### Step 1: 爬蟲抓取Boston房價
#### Step 2: Preprocessing : train test split
#### Step 3: Build Model using Lasso
#### Step 4: Evaluation: MSE, MAE, R2 metrics 的意義, overfit and underfit 的判斷（畫出 training, test curve）, 優化模型 optuna
#### modify step 4 by using Lasso to select the features X_order with different number of variables according to the importance, Use a table to list the number of variable, the name of the variables used in each model, also list the associated RMSE and R2. Plot the RMSE, R2 against the number of variables 
#### Step 5 that includes feature selection using three state-of-the-art feature selection schemes (Mutual Information, Recursive Feature Elimination, and SelectKBest), and then creates a table (align content to the left) to list the features used in those models with different numbers of features according to the importance order from left to right
#### Step 6 plot the RMSE results for each algorithm
======================================================================
# 利用 CRISP-DM 模板解決波士頓房價預測的多元回歸問題(取出重要特徵)
## 1. 商業理解 (Business Understanding)
- **目標**: 預測波士頓地區的房價中位數 (`medv`)。
- **問題定義**: 確定影響房價的主要因素，並開發一個準確的預測模型。

## 2. 數據理解 (Data Understanding)

### 代碼
```python
# Step 1: 數據收集 mark as step1_v2
import pandas as pd

# 直接讀取 GitHub 上的 CSV 文件
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

print(data.head())

```

## 3. 數據準備 (Data Preparation)

### 代碼
```python
# Step 2: 數據準備 mark as step2_v2
from sklearn.model_selection import train_test_split

# 分割特征与标签（medv 是我们的目标变量，即房价中位数）
X = data.drop('medv', axis=1)  # 去除目标变量，剩下的为特征变量
y = data['medv']  # 目标变量

# 使用 80% 数据作为训练集，20% 作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

```

## 4. 建模 (Modeling)

### 代碼
```python
# Step 3: 建立模型 mark as step3_v2
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# 进行标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建 Lasso 模型
lasso = Lasso(alpha=0.1)  # alpha 是正则化强度参数
lasso.fit(X_train_scaled, y_train)

```
## 5. 評估模型 (Evaluation)

### 代碼
```python
# Step 4: 評估模型 mark as step4_v2
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 计算训练集和测试集的评估指标
y_train_pred = lasso.predict(X_train_scaled)
y_test_pred = lasso.predict(X_test_scaled)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"训练集 RMSE: {rmse_train}, 测试集 RMSE: {rmse_test}")
print(f"训练集 R²: {r2_train}, 测试集 R²: {r2_test}")

# 画出训练和测试的曲线
train_sizes = np.arange(1, X_train.shape[1] + 1)
train_rmses = []
test_rmses = []
train_r2s = []
test_r2s = []

for i in train_sizes:
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_scaled[:, :i], y_train)
    y_train_pred = lasso.predict(X_train_scaled[:, :i])
    y_test_pred = lasso.predict(X_test_scaled[:, :i])
    
    train_rmses.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    test_rmses.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    train_r2s.append(r2_score(y_train, y_train_pred))
    test_r2s.append(r2_score(y_test, y_test_pred))

# 绘制RMSE和R²曲线
plt.figure(figsize=(12, 6))

# RMSE
plt.subplot(1, 2, 1)
plt.plot(train_sizes, train_rmses, label='Train RMSE', marker='o')
plt.plot(train_sizes, test_rmses, label='Test RMSE', marker='o')
plt.xlabel('Number of Features')
plt.ylabel('RMSE')
plt.title('RMSE vs Number of Features')
plt.legend()

# R²
plt.subplot(1, 2, 2)
plt.plot(train_sizes, train_r2s, label='Train R²', marker='o')
plt.plot(train_sizes, test_r2s, label='Test R²', marker='o')
plt.xlabel('Number of Features')
plt.ylabel('R²')
plt.title('R² vs Number of Features')
plt.legend()

plt.tight_layout()
plt.show()

```

## 5.1 特徵選擇

```python
# Step 4.1: 評估模型 mark as step4.1_v2
from sklearn.feature_selection import mutual_info_regression, RFE, SelectKBest, f_regression

# 初始化 DataFrame 来存储结果
results_features_df = pd.DataFrame(columns=['Number of Features', 'MI Features', 'RFE Features', 'SelectKBest Features'])

# 遍历不同数量的特征
for num_features in range(1, X_train.shape[1] + 1):
    # 使用互信息进行特征选择
    mi_selector = SelectKBest(score_func=mutual_info_regression, k=num_features)
    mi_selector.fit(X_train, y_train)
    mi_importance = mi_selector.scores_
    mi_indices = np.argsort(mi_importance)[::-1][:num_features]
    mi_selected_features = X_train.columns[mi_indices].tolist()

    # 使用递归特征消除 (RFE)
    rfe_selector = RFE(estimator=Lasso(alpha=0.1), n_features_to_select=num_features)
    rfe_selector.fit(X_train, y_train)
    rfe_selected_features = X_train.columns[rfe_selector.support_].tolist()

    # 使用 SelectKBest 与 f_regression
    skb_selector = SelectKBest(score_func=f_regression, k=num_features)
    skb_selector.fit(X_train, y_train)
    skb_importance = skb_selector.scores_
    skb_indices = np.argsort(skb_importance)[::-1][:num_features]
    skb_selected_features = X_train.columns[skb_indices].tolist()

    # 将结果添加到 DataFrame
    results_features_df = results_features_df.append(
        {
            'Number of Features': num_features,
            'MI Features': ', '.join(mi_selected_features),
            'RFE Features': ', '.join(rfe_selected_features),
            'SelectKBest Features': ', '.join(skb_selected_features)
        }, ignore_index=True)

# 设置显示选项以左对齐内容
pd.set_option('display.colheader_justify', 'left')

# 打印结果表格
print(results_features_df)

```

## 6 繪製RMSE

```python
# 绘制不同特征选择方法的RMSE结果
rmse_results = {}

# 计算RMSE
for index, row in results_features_df.iterrows():
    features = row['MI Features'].split(', ')  # 选取MI的特征
    if features:
        X_train_selected = X_train[features]
        X_test_selected = X_test[features]
        
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train_selected, y_train)
        
        y_test_pred = lasso.predict(X_test_selected)
        rmse_results[row['Number of Features']] = np.sqrt(mean_squared_error(y_test, y_test_pred))

# 绘制RMSE
plt.figure(figsize=(10, 5))
plt.plot(rmse_results.keys(), rmse_results.values(), marker='o')
plt.xlabel('Number of Features')
plt.ylabel('RMSE')
plt.title('RMSE vs Number of Features Selected using MI')
plt.grid()
plt.show()

```


## 7. 部署 (Deployment)

### 代碼
```python
# Step 6: 部屬 mark as  step6_v1
import joblib

# 保存最終模型
joblib.dump(final_lasso, 'lasso_model.pkl')

# 加載模型（在需要時使用）
model = joblib.load('lasso_model.pkl')
```
