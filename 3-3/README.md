# 2D dataset 分布在feature plane上非圓形
## Prompt
### Generate Random Points: Create 600 random points in a 2D Gaussian distribution centered at (0, 0) with a specified variance.
### Calculate Distances: Compute the distance of each point from the origin (0, 0).
### Classify Points: Use an adjustable threshold to classify points into two classes:
### Class 0 if the distance is less than the threshold.
### Class 1 if the distance is greater than or equal to the threshold.
### Define Gaussian Function: Create a Gaussian function to determine the height (z-coordinate) of each point in the scatter plot.
### Visualize Data: Use Plotly to create a 3D scatter plot and display the hyperplane that separates the two classes.

## Output
![image](https://github.com/user-attachments/assets/8727c1d5-5d92-4ed0-9c78-26d35727d369)

