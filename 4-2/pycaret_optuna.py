import pandas as pd
from pycaret.classification import *
import optuna

# 1. Load training and test datasets
train_data = pd.read_csv('./AIOT/Pycarat/train.csv')
test_data = pd.read_csv('./AIOT/Pycarat/test.csv')

# 2. Feature engineering
def feature_engineering(df):
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Age'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 80], labels=['Child', 'Teenager', 'Adult', 'Middle-Aged', 'Senior'])
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    return df

train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)

# 3. Initialize PyCaret
clf = setup(data=train_data, target='Survived', session_id=123)

# 4. Define the Optuna objective function
def objective(trial):
    # Randomly choose a model
    model_name = trial.suggest_categorical('model', ['dt', 'rf', 'lr'])
    
    # Create the model
    model = create_model(model_name)
    
    # Tune the model's hyperparameters
    tuned_model = tune_model(model, n_iter=10)
    
    # Use cross-validation to evaluate model performance
    score = pull().iloc[0]['Accuracy']  # Change 'Accuracy' if needed
    
    return score

# 5. Execute Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Display best parameters and score
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: {}".format(trial.params))

# 6. Retrain final model with best parameters
best_model_name = trial.params['model']
final_model = create_model(best_model_name)
tuned_final_model = tune_model(final_model)

# 7. Make predictions on test data
predictions = predict_model(tuned_final_model, data=test_data)

# Inspect the predictions DataFrame to find the correct column name for predictions
print(predictions.head())  # Check what columns are available

# 8. Save predictions to CSV file
# Adjust this line based on what you find in the predictions DataFrame
if 'Label' in predictions.columns:
    predictions[['PassengerId', 'Label']].rename(columns={'Label': 'Survived'}).to_csv('predictions.csv', index=False)
elif 'prediction_label' in predictions.columns:
    predictions[['PassengerId', 'prediction_label']].rename(columns={'prediction_label': 'Survived'}).to_csv('predictions.csv', index=False)
else:
    print("No suitable prediction column found.")

print("Predictions saved to predictions.csv")