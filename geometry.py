import numpy as np
import matplotlib.pyplot as plt
import logging
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def calculate_angles(p1, p2):
    delta = p2 - p1
    r = np.linalg.norm(delta)
    theta = np.arccos(delta[2] / r)  # elevation angle
    phi = np.arctan2(delta[1], delta[0])  # azimuth angle
    return np.degrees(theta), np.degrees(phi)

def angle_between_vectors(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure cos_theta is within valid range for arccos
    return np.degrees(np.arccos(cos_theta))

def check_coplanarity(points):
    p0 = points[0]
    v1 = points[1] - p0
    v2 = points[2] - p0
    v3 = points[3] - p0
    matrix = np.array([v1, v2, v3])
    volume = np.linalg.det(matrix)
    return np.isclose(volume, 0)

def check_clashes(bend_points, inlet_points, radius=100):
    num_points = len(inlet_points)
    bottom_bend_points = bend_points[:num_points * 3].reshape(num_points, 3)
    top_bend_points = bend_points[num_points * 3:].reshape(num_points, 3)
    for i in range(len(bottom_bend_points)):
        for j in range(i + 1, len(bottom_bend_points)):
            if distance(bottom_bend_points[i], bottom_bend_points[j]) < radius:
                logging.warning(f"Clash detected between bottom bend points {i} and {j}")
                return True
            if distance(top_bend_points[i], top_bend_points[j]) < radius:
                logging.warning(f"Clash detected between top bend points {i} and {j}")
                return True
    return False

def check_for_clashes(point1_start, point1_end, point2_start, point2_end, min_distance=150):
    """Check for clashes between two line segments and apply a penalty if they are too close."""
    def segment_distance(p1, p2, q1, q2):
        """Calculate the shortest distance between two line segments."""
        u = p2 - p1
        v = q2 - q1
        w = p1 - q1
        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w)
        e = np.dot(v, w)
        D = a * c - b * b
        sc = D
        tc = D

        if D < 1e-7:
            sc = 0.0
            tc = (b > c) * d / b if b > c else e / c
        else:
            sc = (b * e - c * d) / D
            tc = (a * e - b * d) / D

        sc = np.clip(sc, 0.0, 1.0)
        tc = np.clip(tc, 0.0, 1.0)

        dP = w + sc * u - tc * v
        return np.linalg.norm(dP)
    
    dist = segment_distance(point1_start, point1_end, point2_start, point2_end)
    if dist < min_distance:
        return (min_distance - dist) ** 2
    return 0.0

def plot_lines_with_bends(ax, inlet_points, bottom_bend_points, top_bend_points, outlet_points):
    ax.clear()
    for i in range(len(inlet_points)):
        inlet_point = inlet_points[i]
        bottom_bend_point = bottom_bend_points[i]
        top_bend_point = top_bend_points[i]
        outlet_point = outlet_points[i]

        plot_lines(ax, inlet_point, bottom_bend_point, top_bend_point, outlet_point, i)
        
        # Plot a 50 mm radius cylinder on the quasi-vertical line segment
        plot_cylinder(ax, bottom_bend_point, top_bend_point, radius=50, color='cyan')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper right')
    plt.draw()
    plt.pause(0.01)

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

def plot_cylinder(ax, start_point, end_point, radius=50, color='c'):
    # Vector along the cylinder's axis
    axis_vector = end_point - start_point
    axis_length = np.linalg.norm(axis_vector)
    axis_vector = axis_vector / axis_length  # Normalize the axis vector

    # Create a circle in the XY plane to be rotated
    theta = np.linspace(0, 2 * np.pi, 30)
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    circle_z = np.zeros_like(circle_x)

    # Stack the circle coordinates into a 3xN array
    circle_points = np.vstack((circle_x, circle_y, circle_z))

    # Create the rotation matrix that aligns the z-axis with the cylinder axis vector
    z_axis = np.array([0, 0, 1])
    if not np.allclose(axis_vector, z_axis):
        v_cross = np.cross(z_axis, axis_vector)
        s = np.linalg.norm(v_cross)
        c = np.dot(z_axis, axis_vector)
        vx = np.array([[0, -v_cross[2], v_cross[1]],
                       [v_cross[2], 0, -v_cross[0]],
                       [-v_cross[1], v_cross[0], 0]])
        rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    else:
        rotation_matrix = np.eye(3)

    # Apply the rotation to the circle points
    rotated_circle = np.dot(rotation_matrix, circle_points)

    # Create the surface points for the cylinder
    z_grid = np.linspace(0, axis_length, 2)
    z_grid, theta_grid = np.meshgrid(z_grid, theta)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)

    # Apply the rotation to each point in the grid
    for i in range(z_grid.shape[0]):
        for j in range(z_grid.shape[1]):
            point = np.array([x_grid[i, j], y_grid[i, j], z_grid[i, j]])
            rotated_point = np.dot(rotation_matrix, point)
            x_grid[i, j] = rotated_point[0] + start_point[0]
            y_grid[i, j] = rotated_point[1] + start_point[1]
            z_grid[i, j] = rotated_point[2] + start_point[2]

    # Plot the cylinder surface
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.3)
