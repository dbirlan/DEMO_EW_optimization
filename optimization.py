import numpy as np
from scipy.optimize import minimize
from geometry import angle_between_vectors
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def optimize_bend_points(inlet_points, initial_bottom_bend_points_list, initial_top_bend_points_list, outlet_points, initial_inlet_to_bottom_directions, initial_top_to_outlet_directions):
    best_result = None
    best_cost = float('inf')
    
    num_points = len(inlet_points)
    total_variables = num_points * 6  # 3 coordinates per point for bottom and top bend points
    
    logging.info(f"Number of points: {num_points}")
    logging.info(f"Expected total variables: {total_variables}")
    
    history = {'total_length': [], 'coplanarity_penalty': [], 'angle_penalty': [], 'length_penalty': [], 'direction_penalty': []}
    
    for initial_bottom_bend_points, initial_top_bend_points in zip(initial_bottom_bend_points_list, initial_top_bend_points_list):
        # Log the shapes of initial guesses
        logging.info(f"Initial bottom bend points shape: {initial_bottom_bend_points.shape}")
        logging.info(f"Initial top bend points shape: {initial_top_bend_points.shape}")
        
        # Initialize x0 as a concatenation of bottom and top bend points
        x0 = np.hstack((initial_bottom_bend_points.flatten(), initial_top_bend_points.flatten()))
        
        # Log the size of x0
        logging.info(f"Size of initial guess x0: {len(x0)}")
        
        # Ensure x0 has the correct size
        if len(x0) != total_variables:
            raise ValueError(f"Initial guess x0 has an incorrect size. Expected {total_variables}, got {len(x0)}.")
        
        bounds = [(None, None)] * len(x0)
        
        # Perform the optimization
        result = minimize(
            optimization_function,
            x0,
            args=(inlet_points, outlet_points, initial_inlet_to_bottom_directions, initial_top_to_outlet_directions, history),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'disp': True}
        )
        
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result
            
    optimized_bottom_bend_points = best_result.x[:num_points * 3].reshape(num_points, 3)
    optimized_top_bend_points = best_result.x[num_points * 3:].reshape(num_points, 3)
    
    return optimized_bottom_bend_points, optimized_top_bend_points, history

def optimization_function(x, inlet_points, outlet_points, initial_inlet_to_bottom_directions, initial_top_to_outlet_directions, history):
    num_points = len(inlet_points)
    total_variables = num_points * 6  # 3 for bottom_bend_points + 3 for top_bend_points

    # Ensure that x has the correct size
    if len(x) != total_variables:
        raise ValueError(f"Input array x has an incorrect size. Expected {total_variables}, got {len(x)}.")

    bottom_bend_points = x[:num_points * 3].reshape(num_points, 3)
    top_bend_points = x[num_points * 3:].reshape(num_points, 3)
    
    total_length = 0.0
    angle_penalty = 0.0
    clash_penalty = 0.0
    direction_penalty = 0.0
    
    for i in range(num_points):
        # Calculate the direction vectors for the current solution
        current_inlet_to_bottom_direction = (bottom_bend_points[i] - inlet_points[i]) / np.linalg.norm(bottom_bend_points[i] - inlet_points[i])
        current_top_to_outlet_direction = (outlet_points[i] - top_bend_points[i]) / np.linalg.norm(outlet_points[i] - top_bend_points[i])
        
        # Fully constrain the directions
        direction_penalty += np.linalg.norm(current_inlet_to_bottom_direction - initial_inlet_to_bottom_directions[i]) ** 2 * 10000
        direction_penalty += np.linalg.norm(current_top_to_outlet_direction - initial_top_to_outlet_directions[i]) ** 2 * 10000
        
        # Calculate length
        length = (
            np.linalg.norm(bottom_bend_points[i] - inlet_points[i]) +
            np.linalg.norm(top_bend_points[i] - bottom_bend_points[i]) +
            np.linalg.norm(outlet_points[i] - top_bend_points[i])
        )
        total_length += length
        
        # Angle penalty: slightly more permissive around 90 degrees
        angle_bottom = angle_between_vectors(
            bottom_bend_points[i] - inlet_points[i],
            top_bend_points[i] - bottom_bend_points[i]
        )
        angle_top = angle_between_vectors(
            top_bend_points[i] - bottom_bend_points[i],
            outlet_points[i] - top_bend_points[i]
        )
        angle_penalty += (max(0, abs(angle_bottom - 90) - 5) ** 2 + max(0, abs(angle_top - 90) - 5) ** 2)
    
    # Calculate clash penalty
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist_ij = np.min([
                np.linalg.norm(bottom_bend_points[i] - bottom_bend_points[j]),
                np.linalg.norm(top_bend_points[i] - top_bend_points[j]),
                np.linalg.norm(inlet_points[i] - inlet_points[j]),
                np.linalg.norm(outlet_points[i] - outlet_points[j])
            ])
            if dist_ij < 100:  # Ensure 100 mm distance
                clash_penalty += (100 - dist_ij) ** 2
    
    # Combine penalties and length into total penalty
    total_penalty = (
        total_length +
        1000 * clash_penalty +  # Penalize clashes heavily
        500 * angle_penalty +    # Penalize deviations from 90 degrees
        10000 * direction_penalty  # Strictly enforce directions
    )
    
    # Log the current penalties for plotting
    history['total_length'].append(total_length)
    history['coplanarity_penalty'].append(0)  # Placeholder if not using coplanarity directly
    history['angle_penalty'].append(angle_penalty)
    history['length_penalty'].append(total_penalty - total_length)
    history['direction_penalty'].append(direction_penalty)
    
    return total_penalty

# Utility function to calculate angles between vectors
def angle_between_vectors(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure within valid range
    theta = np.arccos(cos_theta)
    return np.degrees(theta)
