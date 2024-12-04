import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Business Understanding
# Objective: Develop a machine learning model to classify iris flowers into different species
# based on their sepal and petal measurements

# 2. Data Understanding
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a DataFrame for easier exploration
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# 3. Data Preparation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Modeling
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Create multiple models for comparison
models = {
    'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Perform cross-validation
cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_results[name] = {
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std()
    }

# Train the best performing model (in this case, we'll use SVM)
best_model = SVC(kernel='rbf', random_state=42)
best_model.fit(X_train_scaled, y_train)

# 5. Evaluation
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score
)

# Predictions
y_pred = best_model.predict(X_test_scaled)

# Evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# 6. Deployment Preparation
# Feature importance (for interpretation)
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

# Visualization function for model results
def visualize_results(X, y, model, scaler, target_names):
    # Apply PCA for 2D visualization
    from sklearn.decomposition import PCA
    
    # Scale the data
    X_scaled = scaler.transform(X)
    
    # Reduce to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot
    plt.figure(figsize=(10, 8))
    for i, target_name in enumerate(target_names):
        plt.scatter(
            X_pca[y == i, 0], 
            X_pca[y == i, 1], 
            label=target_name
        )
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.title('Iris Classification Visualization')
    plt.show()

# Run visualizations
plot_feature_importance(best_model, feature_names)
visualize_results(X, y, best_model, scaler, target_names)

# Optional: Save the model
import joblib
joblib.dump(best_model, 'iris_classification_model.pkl')
joblib.dump(scaler, 'iris_scaler.pkl')

print("\nCRISP-DM Workflow Complete!")
print(f"Model Saved: iris_classification_model.pkl")
print(f"Scaler Saved: iris_scaler.pkl")
