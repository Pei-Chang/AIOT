# Step 1: Import necessary libraries and fetch the dataset using a web crawler
import pandas as pd
import requests
from io import StringIO

# Define the URL to fetch the dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

# Fetch the CSV content from the URL
response = requests.get(url)
data_content = response.content.decode('utf-8')

# Convert CSV content to a pandas DataFrame
boston_df = pd.read_csv(StringIO(data_content))

# Print a summary of the dataset
print(boston_df.head())
print(boston_df.info())
print(boston_df.describe())

from sklearn.model_selection import train_test_split

# Step 2: Prepare X, Y using Train-Test Split
X = boston_df.drop('medv', axis=1)  # Features
y = boston_df['medv']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Continue with the rest of the CRISP-DM steps...
print(X.shape)
X.columns

from sklearn.linear_model import Lasso

# Step 3: Build Model using Lasso
lasso_model = Lasso(alpha=0.1)  # You can adjust the alpha value as needed

# Train the Lasso model
lasso_model.fit(X_train, y_train)

# Continue with the rest of the CRISP-DM steps...

# Step 4: Evaluate Model with Feature Importance Ranking from Lasso and Create Table and Plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Ensure you have X_train, X_test, y_train, y_test

# Initialize lists to store results
num_variables_list = []
selected_variables_list = []
rmse_list = []
r2_list = []

# Train Lasso model to select important features
lasso_model = Lasso(alpha=1.0)  # You can adjust the alpha parameter as needed
lasso_model.fit(X_train, y_train)

# Get the coefficients of the Lasso model
coefficients = lasso_model.coef_

# Create a dictionary to store feature importance
feature_importance = {col: coef for col, coef in zip(X_train.columns, coefficients)}

# Sort features by their importance (absolute value of coefficients)
sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

# Iterate over different numbers of variables
for num_vars in range(1, len(sorted_features) + 1):
    # Extract the names of selected features in order of importance
    selected_feature_names = [feature[0] for feature in sorted_features[:num_vars]]

    # Train Lasso model on the selected features
    lasso_model_selected = Lasso(alpha=1.0)
    lasso_model_selected.fit(X_train[selected_feature_names], y_train)

    # Predict target variable for testing dataset using selected features
    y_test_pred_selected = lasso_model_selected.predict(X_test[selected_feature_names])

    # Calculate RMSE and R2 for testing predictions
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_selected))
    r2 = r2_score(y_test, y_test_pred_selected)

    # Append results to lists
    num_variables_list.append(num_vars)
    selected_variables_list.append(', '.join(selected_feature_names))
    rmse_list.append(rmse)
    r2_list.append(r2)

# Create a DataFrame to store results
results_df = pd.DataFrame({
    'Number of Variables': num_variables_list,
    'Selected Variables': selected_variables_list,
    'RMSE': rmse_list,
    'R2': r2_list
})

# Print the results table
print(results_df)

# Plot RMSE and R2 against the number of variables
plt.figure(figsize=(10, 6))
plt.plot(num_variables_list, rmse_list, label='RMSE')
plt.plot(num_variables_list, r2_list, label='R2')
plt.xlabel('Number of Variables')
plt.ylabel('Score')
plt.title('RMSE and R2 vs Number of Variables')
plt.legend()
plt.grid()
plt.show()

# Continue with the remaining steps of the CRISP-DM process

# Step 5: Feature Selection using Different Schemes and Create Left-Aligned Table

from sklearn.feature_selection import mutual_info_regression, RFE, SelectKBest, f_regression

# Initialize a list to store results
results_features_list = []

# Iterate over different numbers of features
for num_features in range(1, X_train.shape[1] + 1):
    # Feature selection using Mutual Information
    mi_selector = SelectKBest(score_func=mutual_info_regression, k=num_features)
    mi_selector.fit(X_train, y_train)
    mi_importance = mi_selector.scores_
    mi_indices = np.argsort(mi_importance)[::-1][:num_features]
    mi_selected_features = X_train.columns[mi_indices].tolist()

    # Feature selection using Recursive Feature Elimination (RFE)
    rfe_selector = RFE(estimator=Lasso(alpha=1.0), n_features_to_select=num_features)
    rfe_selector.fit(X_train, y_train)
    rfe_selected_features = X_train.columns[rfe_selector.support_].tolist()

    # Feature selection using SelectKBest with f_regression
    skb_selector = SelectKBest(score_func=f_regression, k=num_features)
    skb_selector.fit(X_train, y_train)
    skb_importance = skb_selector.scores_
    skb_indices = np.argsort(skb_importance)[::-1][:num_features]
    skb_selected_features = X_train.columns[skb_indices].tolist()

    # Append results to the list
    results_features_list.append({
        'Number of Features': num_features,
        'MI Features': ', '.join(mi_selected_features),
        'RFE Features': ', '.join(rfe_selected_features),
        'SelectKBest Features': ', '.join(skb_selected_features)
    })

# Create a DataFrame from the list of results
results_features_df = pd.DataFrame(results_features_list)

# Set display options to align content to the left
pd.set_option('display.colheader_justify', 'left')

# Print the results table
print(results_features_df)

# Continue with the remaining steps of the CRISP-DM process

import matplotlib.pyplot as plt

# RMSE results for each algorithm
mi_rmse = [7.8695, 6.1462, 5.2224, 4.7998, 4.7647, 4.6032, 4.6351, 4.5191, 4.5201, 4.5561, 4.5684, 4.5077, 4.4841, 4.4859, 4.5477]
rfe_rmse = [7.8695, 6.0133, 5.2166, 4.8195, 4.7840, 4.6325, 4.6557, 4.5177, 4.5424, 4.5524, 4.5857, 4.5280, 4.4999, 4.5041, 4.5041]
skb_rmse = [7.8695, 6.1659, 6.1462, 5.3052, 4.8014, 4.7872, 4.6144, 4.5866, 4.5816, 4.5025, 4.5212, 4.5330, 4.5574, 4.5199, 4.5496]

# Number of features
num_features = list(range(1, 16))

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(num_features, mi_rmse, marker='o', label='Mutual Information')
plt.plot(num_features, rfe_rmse, marker='d', label='Recursive Feature Elimination')
plt.plot(num_features, skb_rmse, marker='s', label='SelectKBest')
plt.xlabel('Number of Features')
plt.ylabel('RMSE')
plt.title('RMSE for Different Feature Selection Algorithms')
plt.legend()
plt.grid()
plt.show()
