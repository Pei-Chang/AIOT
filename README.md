# Linear Regression Problem Solving Steps With Streamlit

## 1. Write Python to Solve Linear Regression Problem

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Create and train the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Make predictions
y_pred = lin_reg.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
```

## 2. Linear Regression with True and Predicted Regression Lines

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate synthetic data
X = 2 * np.random.rand(100, 1)  # Independent variable
y_true = 4 + 3 * X              # True line equation
y = y_true + np.random.randn(100, 1)  # Add noise to the true line

# Train the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Predict the values
y_pred = lin_reg.predict(X)

# Plot true vs predicted lines
plt.scatter(X, y, color='blue', label='Data points')  # Scatter plot for noisy data points
plt.plot(X, y_true, color='green', label='True line')  # Plot the true line in green
plt.plot(X, y_pred, color='red', label='Predicted line')  # Plot the predicted regression line in red
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression: True vs Predicted Line')
plt.show()
```

## 3. CRISP-DM Process for Solving a Linear Regression Problem

### 1. Business Understanding
- **Objective**: Build a linear regression model to predict values of `y` based on an independent variable `X`. We will compare the predicted regression line with the true regression line.

### 2. Data Understanding
- **Data Source**: Synthetic data generated using the formula `y_true = a * X + b` with added noise to simulate real-world variability.
- **Independent Variable (X)**: Random values between 0 and 2.
- **Dependent Variable (y)**: Calculated using the true relationship `y_true = 4 + 3 * X` and adding random noise to simulate variance in data.

### 3. Data Preparation
- **Steps**:
  1. Generate independent variable `X`.
  2. Compute `y_true` using the true relationship.
  3. Add random noise to create `y`, which simulates the actual observations.

### 4. Modeling
- **Linear Regression**: Use Scikit-learn's `LinearRegression` class to fit a regression model to the noisy data.
- **Modeling Approach**: The goal is to minimize the mean squared error between predicted values and actual data points.

### 5. Evaluation
- **Metrics**: 
  - **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and true values.
  - **R² Score**: Indicates the proportion of variance explained by the model.
  
  The model will be evaluated based on how well the predicted regression line matches the true line.

### 6. Deployment
- **Deployment Method**: Display results graphically by plotting the true regression line, the predicted line, and the noisy data points using Matplotlib.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Data Preparation
np.random.seed(42)
X = 2 * np.random.rand(100, 1)             # Generate random X values
y_true = 4 + 3 * X                        # True line: y_true = 4 + 3 * X
y = y_true + np.random.randn(100, 1)       # Add noise to simulate real data

# Step 2: Modeling (Train the linear regression model)
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Step 3: Make Predictions
y_pred = lin_reg.predict(X)

# Step 4: Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Step 5: Visualize the results
plt.scatter(X, y, color='blue', label='Data points')  # Noisy data points
plt.plot(X, y_true, color='green', label='True line')  # True regression line
plt.plot(X, y_pred, color='red', label='Predicted line')  # Predicted regression line
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression: True vs Predicted Line')
plt.show()
```

## 4. Combine All Figures in One, Show Regression Line in Red
### Modify the code to set y=a\*X+50+c\*np.random.randn(n,1),where a is -10 to 10, c is 0 to 100, n is number of points 10 to 500, allow user to input a, c, n as slider.  convert this code to streamlit(don't use matplotlib)

```python
import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit App Layout
st.title("Linear Regression with User-Defined Parameters")

# Step 1: Get User Input for `a`, `c`, and n
st.sidebar.header("Set Parameters")
a = st.sidebar.slider("Select slope (a):", -10.0, 10.0, 3.0)  # Slider for a (-10 to 10)
c = st.sidebar.slider("Select random noise multiplier (c):", 0.0, 100.0, 10.0)  # Slider for c (0 to 100)
n = st.sidebar.slider("Select number of data points (n):", 10, 500, 100)  # Slider for n (10 to 500)

# Step 2: Data Understanding - Generating synthetic data
st.header("Generated Data")
st.write(f"Generating synthetic data with equation: `y = {a} * X + 50 + {c} * random_noise`")

# Generate synthetic data based on user input
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(n, 1)  # Generate n random data points for X
noise = np.random.rand(n, 1)  # Random noise
y_true = a * X + 50 + c * noise  # Modified equation

# Combine into a DataFrame for Altair plotting
df = pd.DataFrame(data={'X': X.flatten(), 'y_true': y_true.flatten()})

# Show the generated data
st.write("Here is the first few rows of the generated data:")
st.write(df.head())  # Display first few rows of the generated data

# Step 3: Data Preparation - No cleaning needed

# Step 4: Modeling - Train a linear regression model
st.header("Linear Regression Model")

# Train the model
model = LinearRegression()
model.fit(X, y_true)  # Fit the model
y_pred = model.predict(X)  # Predicted values

# Add predictions to the dataframe for plotting
df['y_pred'] = y_pred

# Step 5: Evaluation
st.subheader("Model Evaluation")

# Calculate evaluation metrics
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
st.write(f"**R-squared (R²)**: {r2:.2f}")

# Step 6: Deployment - Visualize the true vs predicted regression lines
st.header("True vs Predicted Regression Line")

# Create an Altair chart for the true data points
true_points = alt.Chart(df).mark_point(color='blue').encode(
    x=alt.X('X', title='X'),
    y=alt.Y('y_true', title='y (True Data)'),
    tooltip=['X', 'y_true']
)


# Plot the predicted regression line
predicted_line = alt.Chart(df).mark_line(color='red').encode(
    x='X',
    y='y_pred',

    tooltip=['X', 'y_pred']
)

# Combine the charts (points and regression line)
combined_chart = true_points + predicted_line

st.altair_chart(combined_chart, use_container_width=True)
```
## 5. Run this command in Terminal: streamlit run LinearRegression.py
#### 5.1 Enter your email
#### 5.2 Success log into streamlit & open the windows of the front-end web
#### **If you face the issue that cannot run Streamlit, Try below code in your terminal and run it again.**
```python
python -m venv venv
```
