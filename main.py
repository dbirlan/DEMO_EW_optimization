import logging
from optimization import optimize_bend_points
from geometry import plot_lines_with_bends, calculate_angles, check_coplanarity
from data import inlet_points, initial_bottom_bend_points, initial_top_bend_points, outlet_points

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

logging.info("Starting optimization...")

optimized_bottom_bend_points, optimized_top_bend_points = optimize_bend_points(
    inlet_points, initial_bottom_bend_points, initial_top_bend_points, outlet_points
)

logging.info("Optimization completed.")
logging.info("Optimized bottom bend points:")
for i, point in enumerate(optimized_bottom_bend_points):
    logging.info(f"Bend Point {i + 1}: {point}")

logging.info("Optimized top bend points:")
for i, point in enumerate(optimized_top_bend_points):
    logging.info(f"Bend Point {i + 1}: {point}")

logging.info("Angles for each segment:")
for i in range(len(inlet_points)):
    theta1, phi1 = calculate_angles(inlet_points[i], optimized_bottom_bend_points[i])
    theta2, phi2 = calculate_angles(optimized_bottom_bend_points[i], optimized_top_bend_points[i])
    theta3, phi3 = calculate_angles(optimized_top_bend_points[i], outlet_points[i])
    logging.info(f"Segment {i + 1}:")
    logging.info(f"  Bottom Line: Theta = {theta1:.2f} degrees, Phi = {phi1:.2f} degrees")
    logging.info(f"  Quasi-Vertical Line: Theta = {theta2:.2f} degrees, Phi = {phi2:.2f} degrees")
    logging.info(f"  Top Line: Theta = {theta3:.2f} degrees, Phi = {phi3:.2f} degrees")

coplanarity_bottom = check_coplanarity(optimized_bottom_bend_points)
coplanarity_top = check_coplanarity(optimized_top_bend_points)
logging.info(f"Bottom bend points are {'coplanar' if coplanarity_bottom else 'not coplanar'}.")
logging.info(f"Top bend points are {'coplanar' if coplanarity_top else 'not coplanar'}.")

plot_lines_with_bends(inlet_points, optimized_bottom_bend_points, optimized_top_bend_points, outlet_points)
