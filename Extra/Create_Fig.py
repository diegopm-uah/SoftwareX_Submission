import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# # --- Setup the Figure ---
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # --- 1. Draw the Sphere ---
# # Create points for a sphere
# u = np.linspace(0, 2 * np.pi, 100) # Azimuthal angle (phi)
# v = np.linspace(0, np.pi, 50)  # Polar angle (theta)
# r = 1.8 # Unit sphere

# x_sphere = r * np.outer(np.cos(u), np.sin(v))
# y_sphere = r * np.outer(np.sin(u), np.sin(v))
# z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v))

# # Plot the sphere wireframe
# ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.4, rstride=5, cstride=5)

# # --- 2. Draw Axes and Target Geometry ---
# # Plot faint axes lines
# ax.quiver(0, 0, 0, 3.4, 0, 0, color='r', alpha=0.45, arrow_length_ratio=0.1)
# ax.quiver(0, 0, 0, 0, 3.25, 0, color='g', alpha=0.45, arrow_length_ratio=0.1)
# ax.quiver(0, 0, 0, 0, 0, 2.75, color='b', alpha=0.45, arrow_length_ratio=0.1)

# # Plot the "Target Geometry" now on the +Z axis
# ax.quiver(0, 0, 0, 2.1, 0, 0, color='black', linewidth=5, arrow_length_ratio=0.2)

# # --- 3. Define and Plot the Conical Restriction ---
# # Define cone width (cw) in degrees
# cw_deg = 55
# cw_rad = np.radians(cw_deg)

# # --- MODIFICATIONS FOR +Z AXIS CONE ---
# # phi: full circle [0, 2*pi]
# phi_cone = np.linspace(0, 2 * np.pi, 50)
# # theta: from 0 (Z-axis) up to the cone width [0, cw]
# theta_cone = np.linspace(0, cw_rad, 50)
# # --------------------------------------

# # Create a meshgrid
# phi_grid, theta_grid = np.meshgrid(phi_cone, theta_cone)

# # Convert patch coordinates to Cartesian (standard spherical to cartesian)
# x_cone = r * np.sin(theta_grid) * np.cos(phi_grid)
# y_cone = r * np.sin(theta_grid) * np.sin(phi_grid)
# z_cone = r * np.cos(theta_grid)

# # Plot the conical patch
# ax.plot_surface(x_cone, y_cone, z_cone, color='blue', alpha=0.5)

# # --- 4. Plot Sample Incidence Directions ---
# n_samples = 80
# # Generate random samples *within* the new intervals
# phi_samples = np.random.uniform(0, 2 * np.pi, n_samples)
# theta_samples = np.random.uniform(0, cw_rad, n_samples)

# r = 1.85

# # Convert sample points to Cartesian
# x_samples = r * np.sin(theta_samples) * np.cos(phi_samples)
# y_samples = r * np.sin(theta_samples) * np.sin(phi_samples)
# z_samples = r * np.cos(theta_samples)

# # Plot the sample points
# ax.scatter(x_samples, y_samples, z_samples, c='red', marker='o', s=18)

# # --- 5. Formatting ---
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# # ax.set_title('Visualization of Conical Sampling Strategy (Centered on +Z)')

# # Set equal aspect ratio and limits
# ax.set_box_aspect([1, 1, 1])
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_zticklabels([])

# # Hide the axis tick labels (the numbers)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_zticklabels([])

# # Set a good viewing angle
# ax.view_init(elev=25, azim=45)

# # --- 6. Create Custom Legend ---
# # Update the legend description
# black_line = mlines.Line2D([], [], color='black', lw=3, label='Target Primary Axis (+X)')
# blue_patch = mpatches.Patch(color='blue', alpha=0.6, label=f'Conical Restriction (cw = {cw_deg}$^\circ$)')
# red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
#                         markersize=6, label='Sample Incidence Directions')

# # Add the custom legend
# # ax.legend(handles=[black_line, blue_patch, red_dot], loc='center left', bbox_to_anchor=(-0.2, 0.2))

# plt.savefig('/home/newfasant/N101-IA/Extra/conical_sampling_visualization_plus_z.png', dpi=300, bbox_inches='tight')
# print("Figure 'conical_sampling_visualization_plus_z.png' saved.")

# -------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# --- Setup the Figure ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# --- 1. Draw the Sphere ---
# Create points for a sphere
u = np.linspace(0, 2 * np.pi, 100) # Azimuthal angle (phi)
v = np.linspace(0, np.pi, 50)  # Polar angle (theta)
r = 1.8 # Unit sphere

x_sphere = r * np.outer(np.cos(u), np.sin(v))
y_sphere = r * np.outer(np.sin(u), np.sin(v))
z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the sphere wireframe
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.43, rstride=5, cstride=5)

# --- 2. Draw Axes and Target Geometry ---
# Plot faint axes lines
ax.quiver(0, 0, 0, 3.4, 0, 0, color='r', alpha=0.45, arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 3.25, 0, color='g', alpha=0.45, arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, 2.75, color='b', alpha=0.45, arrow_length_ratio=0.1)

# Plot the "Target Geometry" as a prominent arrow on the +x axis
ax.quiver(0, 0, 0, 2.25, 0, 0, color='black', linewidth=6, arrow_length_ratio=0.3)

# --- 3. Define and Plot the Conical Restriction ---
# Define cone width (cw) in degrees
cw_deg = 55
cw_rad = np.radians(cw_deg)

# Define the intervals as per the description
# phi: [-cw, cw]
# theta: [90-cw, 90+cw] (or [pi/2 - cw, pi/2 + cw] in radians)
phi_cone = np.linspace(-cw_rad, cw_rad, 50)
theta_cone = np.linspace(np.pi/2 - cw_rad, np.pi/2 + cw_rad, 50)

# Create a meshgrid
phi_grid, theta_grid = np.meshgrid(phi_cone, theta_cone)

# Convert patch coordinates to Cartesian
x_cone = r * np.sin(theta_grid) * np.cos(phi_grid)
y_cone = r * np.sin(theta_grid) * np.sin(phi_grid)
z_cone = r * np.cos(theta_grid)

# Plot the conical patch
ax.plot_surface(x_cone, y_cone, z_cone, color='blue', alpha=0.4)

# --- 4. Plot Sample Incidence Directions ---
n_samples = 65
# Generate random samples *within* the intervals
phi_samples = np.random.uniform(-cw_rad, cw_rad, n_samples)
theta_samples = np.random.uniform(np.pi/2 - cw_rad, np.pi/2 + cw_rad, n_samples)

r = 1.835

# Convert sample points to Cartesian
x_samples = r * np.sin(theta_samples) * np.cos(phi_samples)
y_samples = r * np.sin(theta_samples) * np.sin(phi_samples)
z_samples = r * np.cos(theta_samples)

# Plot the sample points
ax.scatter(x_samples, y_samples, z_samples, c='red', marker='o', s=15)

# --- 5. Formatting ---
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_title('Visualization of Conical Sampling Strategy')

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1]) # Important for 3D plots
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Set a good viewing angle
ax.view_init(elev=20, azim=30)

# --- 6. Create Custom Legend ---
# Create proxy artists for the legend
black_line = mlines.Line2D([], [], color='black', lw=3, label='Target Primary Axis (+x)')
blue_patch = mpatches.Patch(color='blue', alpha=0.6, label=f'Front Conical Restriction (cw = {cw_deg}$^\circ$)')
red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=6, label='Sample Incidence Directions')

# Add the custom legend
# ax.legend(handles=[black_line, blue_patch, red_dot], loc='center left', bbox_to_anchor=(-0.2, 0.2))

plt.savefig('/home/newfasant/N101-IA/Extra/conical_sampling_visualization.png', dpi=300, bbox_inches='tight')
print("Figure 'conical_sampling_visualization.png' saved.")

plt.close("all") # Close the figure to free memory