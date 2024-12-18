# Step 1 數據收集 mark as step1_v1
import pandas as pd

# 直接讀取 GitHub 上的 CSV 文件
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

# Step 2 數據準備 mark as step2_v1
from sklearn.model_selection import train_test_split

# 分割特徵與標籤（medv 是我們的目標變量，即房價中位數）
X = data.drop('medv', axis=1)  # 去除目標變量，剩下的為特徵變量
y = data['medv']  # 目標變量

# 使用 80% 數據作為訓練集，20% 作為測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"訓練集大小: {X_train.shape}, 測試集大小: {X_test.shape}")

# Step 3 建立模型 mark as step3_v1
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# 進行標準化處理，將特徵縮放到同一尺度上
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 創建 Lasso 模型，並訓練它
lasso = Lasso(alpha=0.1)  # alpha 是正則化強度參數，數值越大，正則化越強
lasso.fit(X_train_scaled, y_train)

# 預測房價
y_train_pred = lasso.predict(X_train_scaled)
y_test_pred = lasso.predict(X_test_scaled)

# Step 4 評估模型 mark as step4_v1
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 計算訓練集和測試集的評估指標
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"訓練集 MSE: {mse_train}, 測試集 MSE: {mse_test}")
print(f"訓練集 MAE: {mae_train}, 測試集 MAE: {mae_test}")
print(f"訓練集 R²: {r2_train}, 測試集 R²: {r2_test}")

import matplotlib.pyplot as plt

# 繪製訓練集和測試集的預測結果
plt.figure(figsize=(10,5))
plt.scatter(y_train, y_train_pred, label='Train', color='blue')
plt.scatter(y_test, y_test_pred, label='Test', color='red')
plt.xlabel('Actual House Price')
plt.ylabel('Predicted House Price')
plt.legend()
plt.title('Predict - Train vs Test')
plt.show()

# Step 5 超參數優化 mark as step5_v1
import optuna  # 確保導入 optuna

# 定義目標函數，Optuna 將嘗試不同的 alpha 參數來最小化 MSE
def objective(trial):
    alpha = trial.suggest_float('alpha', 0.001, 10.0)  # 調整 alpha 範圍
    
    # 使用新的 alpha 創建 Lasso 模型
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)
    
    # 在測試集上評估模型
    y_test_pred = lasso.predict(X_test_scaled)
    mse_test = mean_squared_error(y_test, y_test_pred)
    
    return mse_test

# 進行超參數搜索
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 獲得最佳 alpha 值
best_alpha = study.best_params['alpha']
print(f"最佳 alpha 值: {best_alpha}")

# 使用最佳 alpha 訓練最終模型
final_lasso = Lasso(alpha=best_alpha)
final_lasso.fit(X_train_scaled, y_train)

# Step 6 Deployment mark as step6_v1
import joblib

# 保存最終模型
joblib.dump(final_lasso, 'lasso_model.pkl')

# 加載模型
model = joblib.load('lasso_model.pkl')
