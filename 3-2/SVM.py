import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Streamlit UI setup
st.title("3D Scatter Plot with Adjustable Hyperplane Threshold")
st.write("This app generates random points around the origin and classifies them with a Gaussian function.")

# Sidebar: User input for hyperplane threshold
threshold = st.slider("Adjust Hyperplane Threshold", min_value=1.0, max_value=20.0, value=4.0, step=0.5)

# Generate 600 random points centered at (0, 0) with variance of 10
np.random.seed(0)
num_points = 600
x1 = np.random.normal(0, 10, num_points)
x2 = np.random.normal(0, 10, num_points)

# Calculate distances and assign labels
distances = np.sqrt(x1**2 + x2**2)
labels = np.where(distances < threshold, 0, 1)

# Define x3 as a Gaussian function of x1 and x2
x3 = np.exp(-0.01 * (x1**2 + x2**2))

# Prepare 3D scatter plot with Plotly
fig = go.Figure()

# Add data points to the plot, color-coded by labels
fig.add_trace(go.Scatter3d(
    x=x1, y=x2, z=x3, 
    mode='markers',
    marker=dict(
        size=5,
        color=labels,  # Color by label
        colorscale=['blue', 'red'],  # Blue for Y=0, Red for Y=1
        opacity=0.7
    ),
    name="Data Points"
))

# Create the adjustable hyperplane
# The hyperplane is set at the user's threshold value for separating the classes.
xx, yy = np.meshgrid(np.linspace(-30, 30, 30), np.linspace(-30, 30, 30))
zz = np.full(xx.shape, np.exp(-0.01 * threshold**2))  # Z-value corresponding to the threshold distance

fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz,
    showscale=False,
    opacity=0.3,
    surfacecolor=np.ones_like(zz),  # Light blue color
    colorscale=[[0, "lightblue"], [1, "lightblue"]],
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
