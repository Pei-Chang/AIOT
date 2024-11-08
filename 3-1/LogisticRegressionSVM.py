

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成資料
np.random.seed(0)
x = np.random.randint(0, 1001, 300)
y = np.where((x >= 500) & (x <= 800), 1, 0)

# 分割訓練集和測試集
x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y, test_size=0.3, random_state=0)

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train)
y_pred_logistic = logistic_model.predict(x_test)

# SVM
svm_model = SVC( probability=True)
svm_model.fit(x_train, y_train)
y_pred_svm = svm_model.predict(x_test)

# 計算準確率
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"Logistic Regression Accuracy: {accuracy_logistic:.2f}")
print(f"SVM Accuracy: {accuracy_svm:.2f}")

# 準備畫 Decision Boundary
x_values = np.linspace(0, 1000, 1000).reshape(-1, 1)  # 0到1000之間的連續數據
y_boundary = svm_model.predict_proba(x_values)[:, 1]

# Logistic Regression 的預測概率
y_prob_logistic = logistic_model.predict_proba(x_values)[:, 1]

# SVM 的預測概率
y_prob_svm = svm_model.predict_proba(x_values)[:, 1]

# 畫圖
plt.figure(figsize=(12, 5))

# Logistic Regression 的預測結果和 Decision Boundary
plt.subplot(1, 2, 1)
plt.scatter(x_test, y_test, color="gray", label="True")
plt.scatter(x_test, y_pred_logistic, color="red",marker="x", label="Predicted")
plt.plot(x_values, y_prob_logistic, color="black", linestyle="--", label="Decision Boundary")
plt.title("Logistic Regression with Decision Boundary")
plt.xlabel("x")
plt.ylabel("y1 (predicted)")
plt.legend()

# SVM 的預測結果和 Decision Boundary
plt.subplot(1, 2, 2)
plt.scatter(x_test, y_test, color="gray", label="True")
plt.scatter(x_test, y_pred_svm, color="green",marker="x", label="Predicted")
plt.plot(x_values, y_boundary, color="black", linestyle="--", label="Decision Boundary")
plt.title("SVM with Decision Boundary")
plt.xlabel("x")
plt.ylabel("y2 (predicted)")
plt.legend()

plt.tight_layout()
plt.show()
