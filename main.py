import numpy as np
import logging
import matplotlib.pyplot as plt
import csv
import win32com.client as win32
from mpl_toolkits.mplot3d import Axes3D
from optimization import optimize_bend_points
from geometry import check_coplanarity, plot_lines_with_bends, angle_between_vectors

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Example input data for multiple lines
from data import inlet_points, initial_bottom_bend_points, initial_top_bend_points, outlet_points

# Log initial inlet points
logging.info("Initial inlet points:")
for i, point in enumerate(inlet_points):
    logging.info(f"Inlet Point {i + 1}: {point}")

# Generate more diverse initial guesses for top bend points
initial_top_bend_points_list = []
for i in range(-4, 5):  # Increase the range from -4 to +4
    initial_top_bend_points_variant = initial_top_bend_points.copy()
    initial_top_bend_points_variant[:, 0] += i * 500  # Increase the range of X coordinates by 500 mm per step
    initial_top_bend_points_list.append(initial_top_bend_points_variant)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

logging.info("Starting optimization...")

# Calculate initial direction vectors
initial_inlet_to_bottom_directions = initial_bottom_bend_points - inlet_points
initial_top_to_outlet_directions = outlet_points - initial_top_bend_points

# Normalize direction vectors
initial_inlet_to_bottom_directions = initial_inlet_to_bottom_directions / np.linalg.norm(initial_inlet_to_bottom_directions, axis=1)[:, np.newaxis]
initial_top_to_outlet_directions = initial_top_to_outlet_directions / np.linalg.norm(initial_top_to_outlet_directions, axis=1)[:, np.newaxis]

# Perform the optimization
optimized_bottom_bend_points, optimized_top_bend_points, history = optimize_bend_points(
    inlet_points, [initial_bottom_bend_points], initial_top_bend_points_list, outlet_points, initial_inlet_to_bottom_directions, initial_top_to_outlet_directions
)

logging.info("Optimization completed.")
logging.info("Optimized bottom bend points:")
for i, point in enumerate(optimized_bottom_bend_points):
    logging.info(f"Bend Point {i + 1}: {point}")

logging.info("Optimized top bend points:")
for i, point in enumerate(optimized_top_bend_points):
    logging.info(f"Bend Point {i + 1}: {point}")

# Log inlet points again to verify they haven't changed
logging.info("Final inlet points (should be unchanged):")
for i, point in enumerate(inlet_points):
    logging.info(f"Inlet Point {i + 1}: {point}")

logging.info("Angles at the bend points:")
for i in range(len(inlet_points)):
    bottom_vector1 = optimized_bottom_bend_points[i] - inlet_points[i]
    bottom_vector2 = optimized_top_bend_points[i] - optimized_bottom_bend_points[i]
    top_vector1 = optimized_top_bend_points[i] - optimized_bottom_bend_points[i]
    top_vector2 = outlet_points[i] - optimized_top_bend_points[i]

    angle_bottom = angle_between_vectors(bottom_vector1, bottom_vector2)
    angle_top = angle_between_vectors(top_vector1, top_vector2)

    logging.info(f"Segment {i + 1}:")
    logging.info(f"  Angle at Bottom Bend Point = {angle_bottom:.2f} degrees")
    logging.info(f"  Angle at Top Bend Point = {angle_top:.2f} degrees")

coplanarity_bottom = check_coplanarity(optimized_bottom_bend_points)
coplanarity_top = check_coplanarity(optimized_top_bend_points)
logging.info(f"Bottom bend points are {'coplanar' if coplanarity_bottom else 'not coplanar'}.")
logging.info(f"Top bend points are {'coplanar' if coplanarity_top else 'not coplanar'}.")

# Export points to CSV for use in CATIA
def export_points_to_csv(file_name, inlet_points, bottom_bend_points, top_bend_points, outlet_points):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Point Type", "X", "Y", "Z"])
        
        for i, point in enumerate(inlet_points):
            writer.writerow([f"Inlet Point {i + 1}", point[0], point[1], point[2]])
        
        for i, point in enumerate(bottom_bend_points):
            writer.writerow([f"Bottom Bend Point {i + 1}", point[0], point[1], point[2]])
        
        for i, point in enumerate(top_bend_points):
            writer.writerow([f"Top Bend Point {i + 1}", point[0], point[1], point[2]])
        
        for i, point in enumerate(outlet_points):
            writer.writerow([f"Outlet Point {i + 1}", point[0], point[1], point[2]])

csv_file_name = "optimized_points.csv"
export_points_to_csv(csv_file_name, inlet_points, optimized_bottom_bend_points, optimized_top_bend_points, outlet_points)

logging.info(f"Exported optimized points to '{csv_file_name}'.")

# Interact with CATIA to create points and lines using the CSV file
def create_geometry_in_catia(csv_file_path):
    # Initialize the CATIA COM object
    catia = win32.Dispatch("CATIA.Application")
    documents = catia.Documents
    part_doc = documents.Add("Part")
    part = part_doc.Part

    hsf = part.HybridShapeFactory
    hybrid_bodies = part.HybridBodies
    if hybrid_bodies.Count == 0:
        hybrid_body = part.HybridBodies.Add()
    else:
        hybrid_body = hybrid_bodies.Item(1)

    # Read the CSV file and create points and lines in CATIA
    points = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            point_type = row[0]
            x = float(row[1])
            y = float(row[2])
            z = float(row[3])
            
            point = hsf.AddNewPointCoord(x, y, z)
            hybrid_body.AppendHybridShape(point)
            points.append(point)

    # Create lines between the points based on correct order
    for i in range(len(inlet_points)):
        inlet_point = points[i]
        bottom_bend_point = points[i + len(inlet_points)]
        top_bend_point = points[i + 2 * len(inlet_points)]
        outlet_point = points[i + 3 * len(inlet_points)]
        
        # Create lines
        line1 = hsf.AddNewLinePtPt(inlet_point, bottom_bend_point)
        line2 = hsf.AddNewLinePtPt(bottom_bend_point, top_bend_point)
        line3 = hsf.AddNewLinePtPt(top_bend_point, outlet_point)
        
        hybrid_body.AppendHybridShape(line1)
        hybrid_body.AppendHybridShape(line2)
        hybrid_body.AppendHybridShape(line3)

    part.Update()
    part_doc.SaveAs("C:\\Users\\birlan\\Desktop\\CAD\\python\\DEMO_EW_optimization\\OptimizedPart6.CATPart")

# Call the function to create geometry in CATIA and save the part
create_geometry_in_catia(csv_file_name)

logging.info("Geometry (points and lines) has been created in CATIA based on the optimized coordinates.")

plot_lines_with_bends(ax, inlet_points, optimized_bottom_bend_points, optimized_top_bend_points, outlet_points)

# Plotting the evolution of constraints
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(history['total_length'], label='Total Length')
plt.xlabel('Iteration')
plt.ylabel('Total Length')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(history['coplanarity_penalty'], label='Coplanarity Penalty')
plt.xlabel('Iteration')
plt.ylabel('Coplanarity Penalty')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(history['angle_penalty'], label='Angle Penalty')
plt.xlabel('Iteration')
plt.ylabel('Angle Penalty')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(history['length_penalty'], label='Length Penalty')
plt.xlabel('Iteration')
plt.ylabel('Length Penalty')
plt.legend()

plt.tight_layout()
plt.show()
