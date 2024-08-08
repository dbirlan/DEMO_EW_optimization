import numpy as np
from scipy.optimize import minimize
from geometry import distance, check_clashes, check_coplanarity
import logging

def optimize_bend_points(inlet_points, initial_bottom_bend_points, initial_top_bend_points, outlet_points):
    def direction_penalty(bend_points, original_bottom_bend_points):
        bottom_bend_points = bend_points[:len(original_bottom_bend_points) * 3].reshape(len(original_bottom_bend_points), 3)
        penalty = 0
        for i in range(len(original_bottom_bend_points)):
            direction_original = original_bottom_bend_points[i] - inlet_points[i]
            direction_new = bottom_bend_points[i] - inlet_points[i]
            penalty += np.linalg.norm(direction_original - direction_new)
        return penalty

    def total_length(bend_points, inlet_points, outlet_points):
        num_points = len(inlet_points)
        bottom_bend_points = bend_points[:num_points * 3].reshape(num_points, 3)
        top_bend_points = bend_points[num_points * 3:].reshape(num_points, 3)
        length = 0
        for i in range(num_points):
            length += distance(inlet_points[i], bottom_bend_points[i])
            length += distance(bottom_bend_points[i], top_bend_points[i])
            length += distance(top_bend_points[i], outlet_points[i])
        return length

    def coplanarity_penalty(bend_points):
        num_points = len(inlet_points)
        bottom_bend_points = bend_points[:num_points * 3].reshape(num_points, 3)
        top_bend_points = bend_points[num_points * 3:].reshape(num_points, 3)
        p0 = bottom_bend_points[0]
        v1 = bottom_bend_points[1] - p0
        v2 = bottom_bend_points[2] - p0
        v3 = bottom_bend_points[3] - p0
        matrix = np.array([v1, v2, v3])
        bottom_volume = np.linalg.det(matrix)
        p0 = top_bend_points[0]
        v1 = top_bend_points[1] - p0
        v2 = top_bend_points[2] - p0
        v3 = top_bend_points[3] - p0
        matrix = np.array([v1, v2, v3])
        top_volume = np.linalg.det(matrix)
        return np.abs(bottom_volume) + np.abs(top_volume)

    def optimization_function(bend_points, *args):
        inlet_points, outlet_points, original_bottom_bend_points = args
        logging.info(f"bend_points shape: {bend_points.shape}")
        if check_clashes(bend_points, inlet_points):
            return np.inf
        length = total_length(bend_points, inlet_points, outlet_points)
        penalty = direction_penalty(bend_points, original_bottom_bend_points)
        coplanarity = coplanarity_penalty(bend_points)
        return length + penalty + 10000 * coplanarity

    # Combine initial guess for bottom and top bend points
    initial_guess = np.hstack((initial_bottom_bend_points.flatten(), initial_top_bend_points.flatten()))
    logging.info(f"Initial guess shape: {initial_guess.shape}")

    # Define bounds for each parameter in the optimization
    num_params = len(initial_guess)
    bounds = [(None, None)] * num_params
    logging.info(f"Bounds shape: {len(bounds)}")

    # Optimize the bottom and top bend points
    result = minimize(
        optimization_function, 
        initial_guess, 
        args=(inlet_points, outlet_points, initial_bottom_bend_points),
        method='L-BFGS-B',
        bounds=bounds
    )

    optimized_bend_points = result.x
    num_points = len(inlet_points)
    optimized_bottom_bend_points = optimized_bend_points[:num_points * 3].reshape(num_points, 3)
    optimized_top_bend_points = optimized_bend_points[num_points * 3:].reshape(num_points, 3)

    return optimized_bottom_bend_points, optimized_top_bend_points

if __name__ == "__main__":
    # Example usage with dummy data
    inlet_points = np.random.rand(18, 3)
    initial_bottom_bend_points = np.random.rand(18, 3) + 10
    initial_top_bend_points = np.random.rand(18, 3) + 20
    outlet_points = np.random.rand(18, 3) + 30

    optimize_bend_points(inlet_points, initial_bottom_bend_points, initial_top_bend_points, outlet_points)
