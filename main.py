import numpy as np
import logging
import matplotlib.pyplot as plt
from optimization import optimize_bend_points
from geometry import plot_lines_with_bends, angle_between_vectors

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Example input data for multiple lines
from data import inlet_points, initial_bottom_bend_points, initial_top_bend_points, outlet_points

# Define initial directions
initial_inlet_to_bottom_directions = [
    (initial_bottom_bend_points[i] - inlet_points[i]) / np.linalg.norm(initial_bottom_bend_points[i] - inlet_points[i])
    for i in range(len(inlet_points))
]
initial_top_to_outlet_directions = [
    (outlet_points[i] - initial_top_bend_points[i]) / np.linalg.norm(outlet_points[i] - initial_top_bend_points[i])
    for i in range(len(inlet_points))
]

# Log initial inlet points
logging.info("Initial inlet points:")
for i, point in enumerate(inlet_points):
    logging.info(f"Inlet Point {i + 1}: {point}")

# Perform the optimization
optimized_bottom_bend_points, optimized_top_bend_points, history = optimize_bend_points(
    inlet_points, [initial_bottom_bend_points], [initial_top_bend_points], outlet_points, 
    initial_inlet_to_bottom_directions, initial_top_to_outlet_directions
)

logging.info("Optimization completed.")
logging.info("Optimized bottom bend points:")
for i, point in enumerate(optimized_bottom_bend_points):
    logging.info(f"Bend Point {i + 1}: {point}")

logging.info("Optimized top bend points:")
for i, point in enumerate(optimized_top_bend_points):
    logging.info(f"Bend Point {i + 1}: {point}")

# Calculate and log angles and lengths for each segment
logging.info("Angles and lengths for each segment before and after optimization:")

for i in range(len(inlet_points)):
    initial_length = (
        np.linalg.norm(initial_bottom_bend_points[i] - inlet_points[i]) +
        np.linalg.norm(initial_top_bend_points[i] - initial_bottom_bend_points[i]) +
        np.linalg.norm(outlet_points[i] - initial_top_bend_points[i])
    )
    optimized_length = (
        np.linalg.norm(optimized_bottom_bend_points[i] - inlet_points[i]) +
        np.linalg.norm(optimized_top_bend_points[i] - optimized_bottom_bend_points[i]) +
        np.linalg.norm(outlet_points[i] - optimized_top_bend_points[i])
    )
    
    initial_bottom_angle = angle_between_vectors(
        initial_bottom_bend_points[i] - inlet_points[i],
        initial_top_bend_points[i] - initial_bottom_bend_points[i]
    )
    initial_top_angle = angle_between_vectors(
        initial_top_bend_points[i] - initial_bottom_bend_points[i],
        outlet_points[i] - initial_top_bend_points[i]
    )
    
    optimized_bottom_angle = angle_between_vectors(
        optimized_bottom_bend_points[i] - inlet_points[i],
        optimized_top_bend_points[i] - optimized_bottom_bend_points[i]
    )
    optimized_top_angle = angle_between_vectors(
        optimized_top_bend_points[i] - optimized_bottom_bend_points[i],
        outlet_points[i] - optimized_top_bend_points[i]
    )
    
    logging.info(f"Segment {i + 1}:")
    logging.info(f"  Initial Length = {initial_length:.2f}, Optimized Length = {optimized_length:.2f}")
    logging.info(f"  Initial Bottom Angle = {initial_bottom_angle:.2f} degrees, Optimized Bottom Angle = {optimized_bottom_angle:.2f} degrees")
    logging.info(f"  Initial Top Angle = {initial_top_angle:.2f} degrees, Optimized Top Angle = {optimized_top_angle:.2f} degrees")

    # Calculate and log lengths for each segment (bottom, quasi-vertical, top)
logging.info("Segment lengths (Bottom, Quasi-Vertical, Top) for each line:")

for i in range(len(inlet_points)):
    bottom_length_initial = np.linalg.norm(initial_bottom_bend_points[i] - inlet_points[i])
    vertical_length_initial = np.linalg.norm(initial_top_bend_points[i] - initial_bottom_bend_points[i])
    top_length_initial = np.linalg.norm(outlet_points[i] - initial_top_bend_points[i])
    
    bottom_length_optimized = np.linalg.norm(optimized_bottom_bend_points[i] - inlet_points[i])
    vertical_length_optimized = np.linalg.norm(optimized_top_bend_points[i] - optimized_bottom_bend_points[i])
    top_length_optimized = np.linalg.norm(outlet_points[i] - optimized_top_bend_points[i])
    
    logging.info(f"Segment {i + 1}:")
    logging.info(f"  Initial Lengths: Bottom = {bottom_length_initial:.2f}, Quasi-Vertical = {vertical_length_initial:.2f}, Top = {top_length_initial:.2f}")
    logging.info(f"  Optimized Lengths: Bottom = {bottom_length_optimized:.2f}, Quasi-Vertical = {vertical_length_optimized:.2f}, Top = {top_length_optimized:.2f}")


# Calculate and log the minimum distance between any two points before and after
initial_min_distance = min(
    np.linalg.norm(initial_bottom_bend_points[i] - initial_bottom_bend_points[j])
    for i in range(len(inlet_points)) for j in range(i + 1, len(inlet_points))
)
optimized_min_distance = min(
    np.linalg.norm(optimized_bottom_bend_points[i] - optimized_bottom_bend_points[j])
    for i in range(len(inlet_points)) for j in range(i + 1, len(inlet_points))
)

logging.info(f"Minimum distance between any two points before optimization: {initial_min_distance:.2f}")
logging.info(f"Minimum distance between any two points after optimization: {optimized_min_distance:.2f}")



# Plot the evolution of constraints
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(history['total_length'], label='Total Length')
plt.xlabel('Iteration')
plt.ylabel('Total Length')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(history['clash_penalty'], label='Clash Penalty')
plt.xlabel('Iteration')
plt.ylabel('Clash Penalty')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(history['angle_penalty'], label='Angle Penalty')
plt.xlabel('Iteration')
plt.ylabel('Angle Penalty')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(history['direction_penalty'], label='Direction Penalty')
plt.xlabel('Iteration')
plt.ylabel('Direction Penalty')
plt.legend()

plt.tight_layout()
plt.show()

# Plot the 3D scatter plot with lines and bends
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_lines_with_bends(ax, inlet_points, optimized_bottom_bend_points, optimized_top_bend_points, outlet_points)
plt.show()  # Ensure the 3D plot is displayed
