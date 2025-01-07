
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle, Circle
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from PIL import Image


class Visualise:

    layer_names = ['polygon', 'line', 'node', 'drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing', 'walkway', 
                   'stop_line', 'carpark_area', 'lane_connector', 'road_divider', 'lane_divider', 'traffic_light']

    non_geometric_polygon_layers = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing',
                                             'walkway', 'stop_line', 'carpark_area']

    layer_colours = {
        'empty': 'white',
        'drivable_area': 'green',
        'road_segment': 'brown',
        'road_block': 'darkred',
        'lane': 'yellow',
        'ped_crossing': 'orange',
        'walkway': 'tan',
        'stop_line': 'red',
        'carpark_area': 'lightblue'
    }

    @staticmethod
    def show_layers(grid):
        """
        Visualizes the grid's layer matrix.

        Displays the layer grid and creates a legend for the different layers.
        """
        # Get the layer matrix
        layer_matrix = np.transpose(grid.get_layer_matrix())

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        flattened_layers = [layer for row in layer_matrix for layer in row]
        unique_layers = {layer for layer in flattened_layers if layer}

        # Create legend for the layer plot
        legend_handles = [Patch(color=Visualise.layer_colours.get(layer, 'white'), label=layer) for layer in unique_layers]

        # Create color matrix for layers
        color_matrix = np.array([
            [to_rgba(Visualise.layer_colours.get(layer, 'white')) for layer in row]
            for row in layer_matrix
        ])

        ax.imshow(color_matrix, origin='lower')
        ax.set_title("Layer Grid")
        ax.legend(handles=legend_handles, loc='upper right')

        plt.title('Layer plot')
        plt.tight_layout()
        #plt.show()

        print('Layer grid visualization complete.')

    @staticmethod
    def plot_layers(grid, path):
        Visualise.show_layers(grid)
        plt.savefig(path)
        plt.close()
        print(f"Layer plot saved as '{path}'.\n")

    @staticmethod
    def show_occ(grid, i):
        """
        Visualizes the grid's occurrence matrix.

        Displays the grid's occurrence data, plotting cells with the 'empty' layer as white 
        and others based on their occurrence value.
        """

        # Get the occurrence matrix for the given iteration
        occ_matrix = np.transpose(grid.get_occ_matrix(i))
        layer_matrix = np.transpose(grid.get_layer_matrix())

        # Create a mask for 'empty' cells
        mask = (layer_matrix == 'empty')

        # Prepare a colormap for occurrence values (excluding 'empty' cells)
        cmap = plt.cm.viridis
        norm = Normalize(vmin=0, vmax=1)  # Normalize occurrence values between 0 and 1

        # Create the plot
        plt.figure(figsize=(10, 8))

        # Plot the occurrence matrix, masking 'empty' cells
        im = plt.imshow(
            np.ma.masked_where(mask, occ_matrix), 
            origin='lower', 
            cmap=cmap, 
            norm=norm
        )

        # Overlay 'empty' cells as white
        plt.imshow(
            np.where(mask, 1, np.nan),  # Masked cells get 1, others are NaN
            origin='lower', 
            cmap=ListedColormap(['white']), 
            interpolation='none'
        )

        # Add colorbar for the occurrence values
        cbar = plt.colorbar(im)
        cbar.set_label('Occurrence Value')

        # Add title and adjust layout
        plt.title(f"Occurrence at iteration {i}")
        plt.tight_layout()
        # plt.show()

        # print('Occurrence visualization complete.')
    
    @staticmethod
    def plot_occ(grid, i, output_folder):
        occ_plot_filename = os.path.join(output_folder, f"occ_plot_iter_{i}.png")
        Visualise.show_occ(grid, i)
        plt.savefig(occ_plot_filename)
        plt.close() 
        print(f"Occurrence plot for iteration {i} saved as '{occ_plot_filename}'.")

    @staticmethod
    def show_risks(grid, index):
        """
        Displays a 2x2 subplot grid for risk matrices: Total Risk, Static Risk, Detect Risk, and Track Risk.

        :param grid: The grid object that holds the risk matrices.
        :param index: The index for the sample (used for dynamic risk calculations).
        """
        # Get matrices
        total_risk_matrix = np.transpose(grid.get_total_risk_matrix(index))
        static_risk_matrix = np.transpose(grid.get_static_risk_matrix())
        detect_risk_matrix = np.transpose(grid.get_detect_risk_matrix(index))
        track_risk_matrix = np.transpose(grid.get_track_risk_matrix(index))

        # Define the figure and 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Adjusted figure size
        axes = axes.flatten()  # Flatten the 2x2 grid into a 1D array for easy iteration

        # Risk matrices and titles
        risk_matrices = {
            "Total Risk": total_risk_matrix,
            "Static Risk": static_risk_matrix,
            "Detect Risk": detect_risk_matrix,
            "Track Risk": track_risk_matrix,
        }

        for i, (title, matrix) in enumerate(risk_matrices.items()):
            ax = axes[i]
            im = ax.imshow(matrix, origin='lower', cmap='viridis', norm=Normalize(vmin=np.min(matrix), vmax=np.max(matrix)))
            ax.set_title(title, fontsize=12)

            # Disable gridlines for each subplot
            ax.grid(False)  # This removes the grid overlay
            
            # Add colorbar for each subplot
            cbar = fig.colorbar(ScalarMappable(norm=im.norm, cmap=im.cmap), ax=ax, shrink=0.8)
            cbar.set_label(title)

        # Add the custom title to the entire figure
        fig.suptitle(f"Risk plots Sample {index}", fontsize=16)
        plt.tight_layout(pad=5.0)  # Increase padding between subplots for better spacing
        #plt.show()

        #print('Risk grid visualization complete.')

    @staticmethod
    def plot_risks(grid, index, output_folder):
        risk_plot_filename = os.path.join(output_folder, f"risk_plot_iter_{index}.png")
        Visualise.show_risks(grid, index)
        plt.savefig(risk_plot_filename)
        plt.close() 
        print(f"Risk plot for sample {index} saved as '{risk_plot_filename}'.")
    
    @staticmethod
    def show_risks_maximised(grid, index, max_total, max_static, max_detect, max_track):
        """
        Displays a 2x2 subplot grid for risk matrices: Total Risk, Static Risk, Detect Risk, and Track Risk.

        :param grid: The grid object that holds the risk matrices.
        :param index: The index for the sample (used for dynamic risk calculations).
        """
        # Get matrices
        total_risk_matrix = np.transpose(grid.get_total_risk_matrix(index))
        static_risk_matrix = np.transpose(grid.get_static_risk_matrix())
        detect_risk_matrix = np.transpose(grid.get_detect_risk_matrix(index))
        track_risk_matrix = np.transpose(grid.get_track_risk_matrix(index))

        # Define the figure and 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Adjusted figure size
        axes = axes.flatten()  # Flatten the 2x2 grid into a 1D array for easy iteration

        # Risk matrices and titles
        risk_matrices = {
            "Total Risk": (total_risk_matrix, max_total),
            "Static Risk": (static_risk_matrix, max_static),  # Assuming static risk shares max_total
            "Detect Risk": (detect_risk_matrix, max_detect),
            "Track Risk": (track_risk_matrix, max_track),
        }

        for i, (title, (matrix, max_value)) in enumerate(risk_matrices.items()):
            ax = axes[i]
            norm = Normalize(vmin=0, vmax=max_value)
            im = ax.imshow(matrix, origin='lower', cmap='viridis', norm=norm)
            ax.set_title(title, fontsize=12)

            # Disable gridlines for each subplot
            ax.grid(False)  # This removes the grid overlay

            # Add colorbar for each subplot
            cbar = fig.colorbar(ScalarMappable(norm=im.norm, cmap=im.cmap), ax=ax, shrink=0.8)
            cbar.set_label(title)

        # Add the custom title to the entire figure
        fig.suptitle(f"Risk plots Sample {index}", fontsize=16)
        # Adjust layout to avoid overlap
        plt.tight_layout(pad=5.0)  # Increase padding between subplots for better spacing
        #plt.show()

        #print('Risk grid visualization complete.')

    @staticmethod
    def plot_risks_maximised(grid, index, maxs, output_folder):
        max_total, max_static, max_detect, max_track = maxs
        risk_plot_filename = os.path.join(output_folder, f"risk_plot_iter_{index}.png")
        Visualise.show_risks_maximised(grid, index, max_total, max_static, max_detect, max_track)  
        plt.savefig(risk_plot_filename)
        plt.close()
        print(f"Risk plot for sample {index} saved as '{risk_plot_filename}'.")

    @staticmethod
    def plot_grid(grid, index, prnt=False):
        """
        Visualizes the grid's layer and risk matrices in a combined layout:
        - Large plot for the layer grid.
        - 2x2 subplot grid for risk plots.
        """
        # Get matrices
        layer_matrix = np.transpose(grid.get_layer_matrix())
        total_risk_matrix = np.transpose(grid.get_total_risk_matrix(index))
        static_risk_matrix = np.transpose(grid.get_static_risk_matrix())
        detect_risk_matrix = np.transpose(grid.get_detect_risk_matrix(index))
        track_risk_matrix = np.transpose(grid.get_track_risk_matrix(index))

        # Define the figure and gridspec layout
        fig = plt.figure(figsize=(18, 12))  # Adjust figure size
        gs = GridSpec(2, 3, figure=fig, width_ratios=[2, 1, 1])  # Define a 2x3 grid layout with custom width ratios

        # Large plot for layer grid (spanning 2 rows and 2 columns)
        ax1 = fig.add_subplot(gs[:, 0])  # Span both rows in the first column (left side)
        
        flattened_layers = [layer for row in layer_matrix for layer in row]
        unique_layers = {layer for layer in flattened_layers if layer}

        # Create legend for layer plot
        legend_handles = [Patch(color=Visualise.layer_colours.get(layer, 'white'), label=layer) for layer in unique_layers]

        # Create color matrix for layers
        color_matrix = np.array([
            [to_rgba(Visualise.layer_colours.get(layer, 'white')) for layer in row]
            for row in layer_matrix
        ])

        ax1.imshow(color_matrix, origin='lower')
        ax1.set_title("Layer Grid")
        ax1.legend(handles=legend_handles, loc='upper right')

        # Risk plots (2x2 grid in the remaining space)
        risk_matrices = {
            "Total Risk": total_risk_matrix,
            "Static Risk": static_risk_matrix,
            "Detect Risk": detect_risk_matrix,
            "Track Risk": track_risk_matrix,
        }

        for i, (title, matrix) in enumerate(risk_matrices.items()):
            # Determine subplot grid position (2x2 right-side subplots)
            ax = fig.add_subplot(gs[i//2, i%2 + 1])  # First row for 0,1 -> second row for 2,3 (right side)
            im = ax.imshow(matrix, origin='lower', cmap='viridis', norm=Normalize(vmin=np.min(matrix), vmax=np.max(matrix)))
            ax.set_title(title)

            # Add colorbar for each subplot
            cbar = fig.colorbar(ScalarMappable(norm=im.norm, cmap=im.cmap), ax=ax)
            cbar.set_label(title)

        plt.tight_layout()
        #plt.show()

        print('Grid visualization complete.')

    @staticmethod
    def save_pointcloud_scatterplot(map, pointcloud, iteration, output_folder,overlay=True, total_size=8, dpi=100):
        """
        Creates and saves a scatter plot of the given point cloud.

        Parameters:
        - pointcloud: list of tuples [(x1, y1), (x2, y2), ...] representing lidar points.
        - iteration: int, the iteration number for naming the file.
        - output_folder: str, the directory where the scatter plot will be saved.
        """
        # Extract map bounds
        grid = map.grid
        ego_pos = map.ego_positions[iteration]

        x_min, x_max , y_min, y_max = grid.patch

        # Calculate aspect ratio
        aspect_ratio = grid.width / grid.length

        # Adjust figsize based on the aspect ratio
        if aspect_ratio >= 1:
            figsize = (total_size, total_size / aspect_ratio)  # Wide map
        else:
            figsize = (total_size * aspect_ratio, total_size)  # Tall map

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Generate the scatter plot
        plt.figure(figsize=figsize,dpi=dpi)  # Adjust figure size as needed

        if overlay:
            # Get the layer matrix from the grid and plot it as the background
            layer_matrix = np.transpose(grid.get_layer_matrix())  # Assuming this gives a matrix of layers
            flattened_layers = [layer for row in layer_matrix for layer in row]
            unique_layers = {layer for layer in flattened_layers if layer}

            # Create the color matrix for the layer grid
            color_matrix = np.array([
                [to_rgba(Visualise.layer_colours.get(layer, 'white')) for layer in row]
                for row in layer_matrix
            ])

            # Display the layer grid as the background
            plt.imshow(color_matrix, origin='lower', extent=[0, grid.width, 0, grid.length])

        plt.scatter(
            [point[0] for point in pointcloud],  # X-coordinates
            [point[1] for point in pointcloud],  # Y-coordinates
            c='black', s=1, marker='.'  # Black points, small size, dot marker
        )

        # Plot the red box at the ego position
        ego_x, ego_y, _ = ego_pos
        ego_x, ego_y = (ego_x-x_min)/grid.res, (ego_y-y_min)/grid.res

        # Add a circle overlay
        circle = Circle((ego_x, ego_y), map.range / grid.res, linewidth=2, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(circle)

        ego_box_size = 0.5  # Define the size of the red box (adjust as needed) In amount of cells covered (currently its a 5x5m box)
        red_box = Rectangle(
            (ego_x - ego_box_size / 2, ego_y - ego_box_size / 2),  # Bottom-left corner
            ego_box_size, ego_box_size,  # Width and height
            linewidth=2, edgecolor='red', facecolor='none'
        )
        plt.gca().add_patch(red_box)  # Add the red box to the plot

        # Set plot background to white
        plt.gca().set_facecolor('white')

        # Set axis limits to map dimensions
        plt.xlim(0, grid.width)
        plt.ylim(0, grid.length)

        # Configure plot aesthetics
        plt.gca().set_aspect('equal', adjustable='box')  # Ensure aspect ratio matches the map
        plt.title(f"Point Cloud Sample {iteration}")
        plt.xlabel("X (Map Coordinates)")
        plt.ylabel("Y (Map Coordinates)")

        # Save the plot
        plot_filename = os.path.join(output_folder, f"pointcloud_iter_{iteration}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Point cloud scatter plot for sample {iteration} saved as '{plot_filename}'.")

    @staticmethod
    def show_lidar_pointcloud_2d(pointcloud, i):

        fig, ax = plt.subplots(figsize=(10, 10))

        # Extract X, Y, Z coordinates from the point cloud
        x_coords = [point[0] for point in pointcloud]
        y_coords = [point[1] for point in pointcloud]

        # Scatter plot of the 3D point cloud
        ax.scatter(x_coords, y_coords, c='black', s=1, marker='.')

        # Setting the labels and title
        ax.set_title(f"2D Point Cloud Sample {i}")
        ax.set_xlabel("X (Global Coordinates)")
        ax.set_ylabel("Y (Global Coordinates)")

        lim = 10
        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        plt.show()


    @staticmethod
    def show_lidar_pointcloud_3d(pointcloud, i):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Extract X, Y, Z coordinates from the point cloud
        x_coords = [point[0] for point in pointcloud]
        y_coords = [point[1] for point in pointcloud]
        z_coords = [point[2] for point in pointcloud]

        # Scatter plot of the 3D point cloud
        ax.scatter(x_coords, y_coords, z_coords, c='black', s=1, marker='.')

        # Setting the labels and title
        ax.set_title(f"3D Point Cloud Sample {i}")
        ax.set_xlabel("X (Global Coordinates)")
        ax.set_ylabel("Y (Global Coordinates)")
        ax.set_zlabel("Z (Global Coordinates)")

        lim = 10
        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        plt.show()
    
    @staticmethod
    def create_gif_from_folder(image_folder, output_gif_path, duration=500):
        """
        Creates and saves a GIF from a folder of images.
        
        Parameters:
        - image_folder: str, path to the folder containing the images.
        - output_gif_path: str, the path where the GIF will be saved.
        - duration: int, the duration for each frame in the GIF in milliseconds (default is 500ms).
        """
        # List all image files in the folder, sorted by file name (for correct ordering)
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

        # Sort files numerically by extracting the numeric part from filenames
        image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        # Load all the images into a list
        images = []
        for image_file in image_files:
            img_path = os.path.join(image_folder, image_file)
            img = Image.open(img_path)
            images.append(img)

        # Save the images as a GIF
        images[0].save(
            output_gif_path, 
            save_all=True, 
            append_images=images[1:], 
            duration=duration, 
            loop=0  # Set loop to 0 for infinite loop, 1 for one-time animation
        )

        print(f"GIF saved as {output_gif_path}")

    @staticmethod
    def plot_avg_risks(maps, output_folder):
        """
        Plots the average risks for two grids on the same plot for comparison.

        :param grid1: The first grid (e.g., constant power simulation)
        :param grid2: The second grid (e.g., variable power simulation)
        :param output_folder: Folder where the plot will be saved
        """
        grid1 = maps[0].grid
        grid2 = maps[1].grid
        plt.figure(figsize=(12, 8))

        # Default matplotlib color cycle (this will be used to maintain consistent colors for the risk factors)
        colors = plt.cm.tab10.colors  # Get the default color cycle from matplotlib (10 distinct colors)

        # Plot risks for grid1 (constant power) with the default color cycle
        plt.plot(grid1.avg_total_risk, label="Total Risk (Constant Power)", marker='+', linestyle='-', color=colors[0])
        plt.plot(grid1.avg_static_risk, label="Static Risk (Constant Power)", marker='o', linestyle='-', color=colors[1])
        plt.plot(grid1.avg_detection_risk, label="Detection Risk (Constant Power)", marker='x', linestyle='-', color=colors[2])
        plt.plot(grid1.avg_tracking_risk, label="Tracking Risk (Constant Power)", marker='s', linestyle='-', color=colors[3])

        # Plot risks for grid2 (variable power) with the same colors as grid1 for consistency
        plt.plot(grid2.avg_total_risk, label="Total Risk (Variable Power)", marker='+', linestyle='--', color=colors[0])
        plt.plot(grid2.avg_static_risk, label="Static Risk (Variable Power)", marker='o', linestyle='--', color=colors[1])
        plt.plot(grid2.avg_detection_risk, label="Detection Risk (Variable Power)", marker='x', linestyle='--', color=colors[2])
        plt.plot(grid2.avg_tracking_risk, label="Tracking Risk (Variable Power)", marker='s', linestyle='--', color=colors[3])

        # Add title and labels
        plt.title("Comparison of Average Risks Between Simulations")
        plt.xlabel("Sample")
        plt.ylabel("Average Risk Value")

        # Add legend to distinguish between grids
        plt.legend()

        # Show grid for better readability
        plt.grid(True)

        # Save the plot
        plot_filename = os.path.join(output_folder, "Average Risks Comparison.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Average risks comparison plot saved as '{plot_filename}'.")


    @staticmethod
    def plot_total_var(var_cons, var_var, title, output_folder):
        plt.figure(figsize=(10, 6))
    
        # Plot constant power simulation occurrence data
        plt.plot(var_cons, label="Constant Power", color="blue", linestyle='-', linewidth=2)

        # Plot variable power simulation occurrence data
        plt.plot(var_var, label="Variable Power", color="red", linestyle='--', linewidth=2)
        
        # Add title and labels
        plt.title(title)
        plt.xlabel("Index")
        
        # Add legend to distinguish between the two lines
        plt.legend(loc='upper right')

        # Show grid for better readability
        plt.grid(True)
        
        # Save the plot
        plot_filename = os.path.join(output_folder, f"{title}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"{title} plot saved as '{plot_filename}'.")


    @staticmethod 
    def plot_avg_occ(maps, output_folder):
        occ_constant = maps[0].grid.avg_occ
        occ_variable = maps[1].grid.avg_occ

        plt.figure(figsize=(10, 6))

        # Plot variable power simulation occurrence data
        plt.plot(occ_variable, label="Variable Power", color="blue", linestyle='-', linewidth=2)
        
        # Plot constant power simulation occurrence data
        plt.plot(occ_constant, label="Constant Power", color="red", linestyle='--', linewidth=2)

        # Add title and labels
        plt.title('Average Occurrence')
        plt.xlabel("Sample")
        plt.ylim(0, 1)

        # Add legend to distinguish between the two lines
        plt.legend(loc='upper right')

        # Show grid for better readability
        plt.grid(True)

        # Save the plot
        plot_filename = os.path.join(output_folder, "Average Occurrence.png")
        plt.savefig(plot_filename)
        plt.close()

        print(f"Average Occurrence plot saved as '{plot_filename}'.")


    @staticmethod
    def show_lidarpointcloud_nusc(map, i):
        
        first = map.scene['first_sample_token']
        last = map.scene['last_sample_token']

        samples = []
        sample = first
        while sample != last:
            samples.append(sample)
            info = map.nusc.get('sample', sample)
            sample = info['next']
        samples.append(last)

        my_sample = map.nusc.get('sample', samples[i])
        map.nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=1, underlay_map=True)

    @staticmethod
    def plot_occ_histogram(map, timestep, output_folder):
        """
        Generate a histogram plot of total_occ_ranges for a specific timestep.

        :param ranges: List or array of range values (e.g., np.linspace(RANGE/10, RANGE, 10)).
        :param total_occ_ranges: 2D list of total occupancy values per range and timestep.
        :param timestep: The timestep (index) to plot the histogram for.
        """
        ranges = map.grid.ranges
        total_occ_ranges = map.grid.avg_occ_ranges
        # Validate timestep
        if not (0 <= timestep < len(total_occ_ranges)):
            raise ValueError(f"Timestep {timestep} is out of bounds. Must be between 0 and {len(total_occ_ranges) - 1}.")

        ranges = np.append(0, ranges)
        # Generate range labels
        range_labels = [f"{ranges[i]:.1f}-{ranges[i+1]:.1f}m" for i in range(len(ranges)-1)]

        # Data for the histogram
        occ_values = total_occ_ranges[timestep]

        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.bar(range_labels, occ_values, color='blue', alpha=0.7)
        plt.ylim(0,1)
        plt.xlabel("Range (meters)")
        plt.ylabel("Average Occurrence per Cell")
        plt.title(f"Average Occurrence per Cell Histogram for Sample {timestep}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plot_filename = os.path.join(output_folder, f"occ_hist_it_{timestep}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Occurrence histogram sample {timestep} saved as '{plot_filename}'.")

    @staticmethod
    def plot_avg_occ_histogram(maps, output_folder):
        """
        Generate a histogram plot of total_occ_ranges for the given maps (constant and variable power).
        
        :param maps: List of map objects. 
                     maps[0] should be constant power, maps[1] should be variable power.
        :param output_folder: Folder to save the generated plot.
        """
        # Extract ranges and avg_occ_ranges from each map (constant power first, variable power second)
        map_constant = maps[0]
        map_variable = maps[1]

        ranges_constant = np.append(0, map_constant.grid.ranges)
        avg_occ_ranges_constant = map_constant.grid.avg_occ_ranges
        range_labels_constant = [f"{ranges_constant[i]:.1f}-{ranges_constant[i+1]:.1f}m" for i in range(len(ranges_constant)-1)]
        avg_occ_values_constant = np.mean(avg_occ_ranges_constant, axis=0)

        ranges_variable = np.append(0, map_variable.grid.ranges)
        avg_occ_ranges_variable = map_variable.grid.avg_occ_ranges
        range_labels_variable = [f"{ranges_variable[i]:.1f}-{ranges_variable[i+1]:.1f}m" for i in range(len(ranges_variable)-1)]
        avg_occ_values_variable = np.mean(avg_occ_ranges_variable, axis=0)

        # Create a plot
        plt.figure(figsize=(10, 6))

        # Set the width for the bars to prevent overlap
        bar_width = 0.35  # You can adjust this to control the distance between the bars

        # Calculate the positions for the bars (offset the variable power bars)
        x_positions_constant = np.arange(len(range_labels_constant))
        x_positions_variable = x_positions_constant + bar_width  # Offset the variable power bars

        # Plot for constant power map (first map)
        plt.bar(x_positions_constant, avg_occ_values_constant, color='blue', alpha=0.7, width=bar_width, label="Constant Power")

        # Plot for variable power map (second map)
        plt.bar(x_positions_variable, avg_occ_values_variable, color='red', alpha=0.7, width=bar_width, label="Variable Power")

        # Add labels and title
        plt.ylim(0, 1)
        plt.xlabel("Range (meters)")
        plt.ylabel("Average Occurrence per Cell")
        plt.title(f"Average Occurrence per Cell Histogram")
        plt.xticks(x_positions_constant + bar_width / 2, range_labels_constant, rotation=45)  # Center the labels between the bars
        plt.tight_layout()

        # Add legend to distinguish between the two simulations
        plt.legend()

        # Save the plot
        plot_filename = os.path.join(output_folder, f"average_total_occ_hist.png")
        plt.savefig(plot_filename)
        plt.close()

        print(f"Average Total Occurrence histogram saved as '{plot_filename}'.")

    @staticmethod
    def plot_power_profile(variable_power, i, output_folder):
        """
        Visualizes the power profile as a histogram for two power profiles.

        Args:
        constant_power (list or np.array): Power values for the constant profile (in watts).
        variable_power (list or np.array): Power values for the variable profile (in watts).
        """
        # Number of cones
        num_cones = len(variable_power)
        constant_power = [64] * num_cones
        
        # Calculate cone angles dynamically
        cone_labels = [f"{j * 360 // num_cones}-{(j + 1) * 360 // num_cones}°" for j in range(num_cones)]
        x = np.arange(num_cones)  # X-axis positions for bars

        # Dynamic bar width
        bar_width = 0.8 / 2  # Divide available space into 2 bars per cone

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot constant power profile
        plt.bar(x - bar_width / 2, constant_power, width=bar_width, label='Constant Power', color='blue')

        # Plot variable power profile
        plt.bar(x + bar_width / 2, variable_power, width=bar_width, label='Variable Power', color='red')

        # Customizing the plot
        plt.xlabel('Cone Angles', fontsize=12)
        plt.ylabel('Power (Watts)', fontsize=12)
        plt.title(f'Power Profile Visualization Sample {i}', fontsize=14)
        plt.xticks(x, cone_labels, fontsize=10)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
       # Save the plot
        plot_filename = os.path.join(output_folder, f"power_profile_it_{i}.png")
        plt.savefig(plot_filename)
        plt.close()

        print(f"Power profile histogram saved as '{plot_filename}'.")

    @staticmethod
    def plot_removed_pointcount(rem_points_cons, rem_points_var, output_folder):

        # Calculate the difference between the two simulations
        difference = [sim1 - sim2 for sim1, sim2 in zip(rem_points_cons, rem_points_var)]

        plt.figure(figsize=(12, 8))

        # Plot removed lidar points for Simulation 1
        plt.plot(rem_points_cons, label="Simulation 1", color="blue", linewidth=2)

        # Plot removed lidar points for Simulation 2
        plt.plot(rem_points_var, label="Simulation 2", color="red", linewidth=2)

        # Plot the difference
        plt.plot(difference, label="Difference", color="green", linestyle=':', linewidth=2)

        # Add title and labels
        plt.title('Removed Lidar Points')
        plt.xlabel("Sample")
        plt.ylabel("Removed Points")

        # Add legend to distinguish between the lines
        plt.legend(loc='upper right')

        # Show grid for better readability
        plt.grid(True)

        # Save the plot
        plot_filename = os.path.join(output_folder, "Removed_Lidar_Points.png")
        plt.savefig(plot_filename)
        plt.close()

        print(f"Removed Lidar Points plot saved as '{plot_filename}'.")