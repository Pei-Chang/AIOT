import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Title and description
st.title("3D Scatter Plot with Adjustable Hyperplane Threshold")
st.write("This app generates random points and classifies them based on their distance from the origin with an adjustable threshold.")

# Sidebar: User input for hyperplane threshold
threshold = st.slider("Adjust Hyperplane Threshold", min_value=1.0, max_value=20.0, value=4.0, step=0.5)

# Step 1: Generate random points
np.random.seed(0)
num_points = 600
mean = 0
variance = 10
x1 = np.random.normal(mean, np.sqrt(variance), num_points)
x2 = np.random.normal(mean, np.sqrt(variance), num_points)

# Step 2: Calculate distances and assign labels based on threshold
distances = np.sqrt(x1**2 + x2**2)
labels = np.where(distances < threshold, 0, 1)  # Classify based on threshold

# Step 3: Define x3 as a Gaussian function of x1 and x2
def gaussian_function(x1, x2):
    return np.exp(-0.01 * (x1**2 + x2**2))

x3 = gaussian_function(x1, x2)

# Prepare the 3D scatter plot with Plotly
fig = go.Figure()

# Add points to the plot, color-coded by label
fig.add_trace(go.Scatter3d(
    x=x1, y=x2, z=x3,
    mode='markers',
    marker=dict(
        size=5,
        color=labels,  # Color by label
        colorscale=['blue', 'red'],  # Blue for Class 0, Red for Class 1
        opacity=0.7
    ),
    name="Data Points"
))

# Create the hyperplane (a flat plane at z = exp(-0.01 * threshold^2))
xx, yy = np.meshgrid(np.linspace(-30, 30, 30), np.linspace(-30, 30, 30))
zz = np.full(xx.shape, np.exp(-0.01 * threshold**2))  # Flat plane at z based on threshold

# Add the hyperplane surface in light gray
fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz,
    showscale=False,
    opacity=0.3,
    surfacecolor=np.ones_like(zz),  # Light gray color
    colorscale=[[0, "lightgray"], [1, "lightgray"]],
    name="Hyperplane"
))

# Update layout for better visualization
fig.update_layout(
    scene=dict(
        xaxis=dict(title="X1"),
        yaxis=dict(title="X2"),
        zaxis=dict(title="X3 (Gaussian Function)"),
    ),
    title=f"3D Scatter Plot with Adjustable Hyperplane (Threshold = {threshold})",
    margin=dict(l=0, r=0, b=0, t=40)
)

# Display the plot in Streamlit
st.plotly_chart(fig)