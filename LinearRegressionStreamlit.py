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