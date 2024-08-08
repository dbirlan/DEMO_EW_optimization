import numpy as np
import matplotlib.pyplot as plt
import logging

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def calculate_angles(p1, p2):
    delta = p2 - p1
    r = np.linalg.norm(delta)
    theta = np.arccos(delta[2] / r)  # elevation angle
    phi = np.arctan2(delta[1], delta[0])  # azimuth angle
    return np.degrees(theta), np.degrees(phi)

def check_coplanarity(points):
    p0 = points[0]
    v1 = points[1] - p0
    v2 = points[2] - p0
    v3 = points[3] - p0
    matrix = np.array([v1, v2, v3])
    volume = np.linalg.det(matrix)
    return np.isclose(volume, 0)

def check_clashes(bend_points, inlet_points, radius=200):
    logging.info(f"bend_points shape: {bend_points.shape}")
    num_points = len(inlet_points)
    bottom_bend_points = bend_points[:num_points * 3].reshape(num_points, 3)
    logging.info(f"bottom_bend_points shape: {bottom_bend_points.shape}")
    top_bend_points = bend_points[num_points * 3:].reshape(num_points, 3)
    logging.info(f"top_bend_points shape: {top_bend_points.shape}")
    for i in range(len(bottom_bend_points)):
        for j in range(i + 1, len(bottom_bend_points)):
            if distance(bottom_bend_points[i], bottom_bend_points[j]) < radius:
                return True
            if distance(top_bend_points[i], top_bend_points[j]) < radius:
                return True
    return False

def plot_lines_with_bends(inlet_points, bottom_bend_points, top_bend_points, outlet_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(inlet_points)):
        inlet_point = inlet_points[i]
        bottom_bend_point = bottom_bend_points[i]
        top_bend_point = top_bend_points[i]
        outlet_point = outlet_points[i]

        plot_lines(ax, inlet_point, bottom_bend_point, top_bend_point, outlet_point, i)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper right')

    plt.show()

def plot_lines(ax, inlet_point, bottom_bend_point, top_bend_point, outlet_point, index):
    # Plot the line segments
    ax.plot([inlet_point[0], bottom_bend_point[0]], [inlet_point[1], bottom_bend_point[1]], [inlet_point[2], bottom_bend_point[2]], 'r-', label='Bottom Lines' if index == 0 else "")
    ax.plot([bottom_bend_point[0], top_bend_point[0]], [bottom_bend_point[1], top_bend_point[1]], [bottom_bend_point[2], top_bend_point[2]], 'g-', label='Quasi-Vertical Lines' if index == 0 else "")
    ax.plot([top_bend_point[0], outlet_point[0]], [top_bend_point[1], outlet_point[1]], [top_bend_point[2], outlet_point[2]], 'b-', label='Top Lines' if index == 0 else "")

    # Plot points
    ax.scatter(inlet_point[0], inlet_point[1], inlet_point[2], color='red', s=50, label='Inlet Point' if index == 0 else "")
    ax.scatter(bottom_bend_point[0], bottom_bend_point[1], bottom_bend_point[2], color='green', s=50, label='Bottom Bend Point' if index == 0 else "")
    ax.scatter(top_bend_point[0], top_bend_point[1], top_bend_point[2], color='blue', s=50, label='Top Bend Point' if index == 0 else "")
    ax.scatter(outlet_point[0], outlet_point[1], outlet_point[2], color='purple', s=50, label='Outlet Point' if index == 0 else "")