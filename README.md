# Optical Routing Optimization

This project aims to optimize the routing of optical lines from given inlet points to outlet points using a series of bend points. The optimization ensures that the routing lines do not intersect and minimizes the total length of the lines while keeping the bend points coplanar.

## Project Structure

The project is divided into four main Python files:

1. **main.py**: This is the entry point of the project. It sets up the data, runs the optimization, and plots the results.
2. **data.py**: Contains the initial data for inlet points, initial bottom bend points, initial top bend points, and outlet points.
3. **geometry.py**: Contains functions related to geometry calculations, such as distance calculation, angle calculation, coplanarity check, and clash detection.
4. **optimization.py**: Contains the optimization function that adjusts the bend points to minimize the total length of the lines and ensure the lines do not intersect.

## Setup

1. Ensure you have Python installed on your system.
2. Install the necessary Python packages:
   ```bash
   pip install numpy matplotlib scipy
   ```

## Usage

1. **Define the data**: Update data.py with the inlet and outlet points, and initial bottom bend points, initial top bend points.
```bash
# data.py
import numpy as np

# Define the inlet points, initial bottom bend points, initial top bend points, and outlet points
inlet_points = np.array([
    [19810.316, 159.068, 561.425],
    # Add the remaining 17 inlet points
], dtype=np.float64)

initial_bottom_bend_points = np.array([
    [22809.926, 188.118, 600.158],
    # Add the remaining 17 initial bottom bend points
], dtype=np.float64)

initial_top_bend_points = np.array([
    [22450, 2400, 2550],
    # Add the remaining 17 initial top bend points
], dtype=np.float64)

outlet_points = np.array([[initial_top_bend_points[i][0] + 5000, initial_top_bend_points[i][1], initial_top_bend_points[i][2]] for i in range(len(initial_top_bend_points))], dtype=np.float64)
```

2. **Run the optimization**: Execute the main.py script to start the optimization process.
```bash
   python main.py
   ```
3. **View the results**: The script will output the optimized bend points and plot the lines. It will also log whether the bottom and top bend points are coplanar.

## What is Being Optimized

The optimization aims to adjust the coordinates of the bottom and top bend points to achieve several objectives and constraints:

## Objectives

1. **Minimize Total Length:** The primary objective is to minimize the total length of the lines. This includes the sum of the lengths of the segments from inlet points to bottom bend points, from bottom bend points to top bend points, and from top bend points to outlet points.
Constraints

2. **Avoid Clashes:** Ensure that the routing lines do not intersect. The check_clashes function checks if the lines are at least a certain distance apart (default is 200 mm).

3. **Maintain Directions:** The direction from inlet points to bottom bend points must be preserved.

4. **Coplanarity of Bend Points:** Both the bottom bend points and top bend points should be coplanar. This is checked using the check_coplanarity function.

## Penalties

1. **Direction Penalty:** A penalty is applied if the direction of the lines from the inlet points to the bottom bend points deviates significantly from the original direction. This is calculated in the direction_penalty function.

2. **Coplanarity Penalty:** A large penalty is applied if the bottom bend points or the top bend points are not coplanar. This is calculated in the coplanarity_penalty function and ensures that the optimizer prioritizes coplanarity.

## Code Explanation

**main.py**
Imports the necessary modules and functions.
Loads the initial data from data.py.
Runs the optimization function from optimization.py.
Logs the optimized bend points and their angles.
Plots the optimized lines.

**data.py**
Contains the initial coordinates for inlet points, bottom bend points, top bend points, and outlet points.

**geometry.py**
distance(p1, p2): Calculates the Euclidean distance between two points.
calculate_angles(p1, p2): Calculates the angles between two points.
check_coplanarity(points): Checks if a set of points are coplanar.
check_clashes(bend_points, inlet_points, radius=200): Checks if any lines intersect or clash.

**optimization.py**
optimize_bend_points(inlet_points, initial_bottom_bend_points, initial_top_bend_points, outlet_points): Optimizes the bend points to minimize the total length of the lines and ensure no intersections.
Contains internal helper functions to calculate direction penalty, total length, and coplanarity penalty.

## Notes

The project assumes that the inlet points and initial bend points are correctly set up in data.py.
The optimization process uses the L-BFGS-B method from SciPy to adjust the bend points.
Ensure that the number of points in initial_bottom_bend_points and initial_top_bend_points matches the number of inlet_points.