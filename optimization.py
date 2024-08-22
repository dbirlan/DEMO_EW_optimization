import numpy as np
from scipy.optimize import minimize
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def optimize_bend_points(inlet_points, initial_bottom_bend_points_list, initial_top_bend_points_list, outlet_points, initial_inlet_to_bottom_directions, initial_top_to_outlet_directions):
    best_result = None
    best_cost = float('inf')
    
    num_points = len(inlet_points)
    total_variables = num_points * 6  # 3 coordinates per point for bottom and top bend points
    
    logging.info(f"Number of points: {num_points}")
    logging.info(f"Expected total variables: {total_variables}")
    
    history = {'total_length': [], 'angle_penalty': [], 'clash_penalty': [], 'direction_penalty': []}
    
    for initial_bottom_bend_points, initial_top_bend_points in zip(initial_bottom_bend_points_list, initial_top_bend_points_list):
        logging.info(f"Initial bottom bend points shape: {initial_bottom_bend_points.shape}")
        logging.info(f"Initial top bend points shape: {initial_top_bend_points.shape}")
        
        x0 = np.hstack((initial_bottom_bend_points.flatten(), initial_top_bend_points.flatten()))
        
        logging.info(f"Size of initial guess x0: {len(x0)}")
        
        if len(x0) != total_variables:
            raise ValueError(f"Initial guess x0 has an incorrect size. Expected {total_variables}, got {len(x0)}.")
        
        bounds = [(None, None)] * len(x0)
        
        result = minimize(
            optimization_function,
            x0,
            args=(inlet_points, outlet_points, initial_inlet_to_bottom_directions, initial_top_to_outlet_directions, history),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50000, 'disp': True, 'gtol': 1e-3, 'ftol': 1e-3}  # Increased maxiter, adjusted tolerances
        )
        
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result
            
    optimized_bottom_bend_points = best_result.x[:num_points * 3].reshape(num_points, 3)
    optimized_top_bend_points = best_result.x[num_points * 3:].reshape(num_points, 3)
    
    return optimized_bottom_bend_points, optimized_top_bend_points, history

def optimization_function(x, inlet_points, outlet_points, initial_inlet_to_bottom_directions, initial_top_to_outlet_directions, history):
    num_points = len(inlet_points)
    total_variables = num_points * 6

    if len(x) != total_variables:
        raise ValueError(f"Input array x has an incorrect size. Expected {total_variables}, got {len(x)}.")

    bottom_bend_points = x[:num_points * 3].reshape(num_points, 3)
    top_bend_points = x[num_points * 3:].reshape(num_points, 3)
    
    total_length = 0.0
    angle_penalty = 0.0
    clash_penalty = 0.0
    direction_penalty = 0.0
    
    for i in range(num_points):
        # Project bottom bend points onto the initial inlet-to-bottom direction
        bottom_bend_points[i] = inlet_points[i] + initial_inlet_to_bottom_directions[i] * np.dot(bottom_bend_points[i] - inlet_points[i], initial_inlet_to_bottom_directions[i])

        # Project top bend points onto the initial top-to-outlet direction
        top_bend_points[i] = outlet_points[i] - initial_top_to_outlet_directions[i] * np.dot(outlet_points[i] - top_bend_points[i], initial_top_to_outlet_directions[i])
        
        # Calculate lengths
        length = (
            np.linalg.norm(bottom_bend_points[i] - inlet_points[i]) +
            np.linalg.norm(top_bend_points[i] - bottom_bend_points[i]) +
            np.linalg.norm(outlet_points[i] - top_bend_points[i])
        )
        total_length += length
        
        # Calculate angle penalties
        angle_bottom = angle_between_vectors(
            bottom_bend_points[i] - inlet_points[i],
            top_bend_points[i] - bottom_bend_points[i]
        )
        angle_top = angle_between_vectors(
            top_bend_points[i] - bottom_bend_points[i],
            outlet_points[i] - top_bend_points[i]
        )
        # Penalty for deviating from the 80-100 degree range, with preference for 90 degrees
        angle_penalty += (
            max(0, abs(angle_bottom - 90) - 10) ** 2 * 1000 + 
            max(0, abs(angle_top - 90) - 10) ** 2 * 1000
        )
        
        # Enforce direction by penalizing deviation from the initial direction vectors
        current_bottom_direction = (bottom_bend_points[i] - inlet_points[i]) / np.linalg.norm(bottom_bend_points[i] - inlet_points[i])
        current_top_direction = (outlet_points[i] - top_bend_points[i]) / np.linalg.norm(outlet_points[i] - top_bend_points[i])
        
        direction_penalty += (
            np.linalg.norm(current_bottom_direction - initial_inlet_to_bottom_directions[i]) ** 2 +
            np.linalg.norm(current_top_direction - initial_top_to_outlet_directions[i]) ** 2
        )
    
    for i in range(num_points):
        for j in range(i + 1, num_points):
            # Check for clashes between the segments
            clash_penalty += check_clashes(inlet_points[i], bottom_bend_points[i], inlet_points[j], bottom_bend_points[j])
            clash_penalty += check_clashes(bottom_bend_points[i], top_bend_points[i], bottom_bend_points[j], top_bend_points[j])
            clash_penalty += check_clashes(top_bend_points[i], outlet_points[i], top_bend_points[j], outlet_points[j])
    
    total_penalty = (
        2000 * total_length +        # Reduced importance on length
        15000 * clash_penalty +      # Enforce clashes more heavily
        5000 * angle_penalty +       # Enforce angle close to 90 degrees but with a range of 80-100
        20000 * direction_penalty    # Strong penalty for direction changes
    )
    
    # Append to history
    history['total_length'].append(total_length)
    history['angle_penalty'].append(angle_penalty)
    history['clash_penalty'].append(clash_penalty)
    history['direction_penalty'].append(direction_penalty)

    # Log every 100 iterations
    if len(history['total_length']) % 100 == 0:
        iteration = len(history['total_length'])
        logging.info(f"Iteration {iteration}: Penalties -> Total Length: {total_length:.2f}, "
                     f"Angle: {angle_penalty:.2f}, Clash: {clash_penalty:.2f}, Direction: {direction_penalty:.2f}")
    
    return total_penalty

def check_clashes(point1_start, point1_end, point2_start, point2_end, min_distance=150):
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

def angle_between_vectors(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure within valid range
    theta = np.arccos(cos_theta)
    return np.degrees(theta)
