# Utility class
from Grid import Grid
from Map import Map
from Risk import Risk
from Visualise import Visualise

import os

class Utility:

    def reinitialize_with_layers(original_grid):
        """
        Reinitialize a Grid object, retaining only the layers information for each cell.
        """
        # Create a new Grid object with the same patch, resolution, and scene length
        new_grid = Grid(
            patch=original_grid.patch,
            resolution=original_grid.res,
            scene_length=original_grid.scene_length,
            RANGE=100  # Reuse or adjust RANGE as needed
        )

        # Iterate through the original grid and copy the `layers` into the new cells
        for x in range(original_grid.width):
            for y in range(original_grid.length):
                original_cell = original_grid.grid[x][y]
                new_grid.grid[x][y].layers = original_cell.layers  # Retain layers
                new_grid.grid[x][y].assign_layer()


        # Copy the `has_assigned_layers` attribute
        new_grid.has_assigned_layers = original_grid.has_assigned_layers

        # Reset grid-level risk and occurrence attributes
        new_grid.avg_total_risk = [0] * original_grid.scene_length
        new_grid.avg_static_risk = [0] * original_grid.scene_length
        new_grid.avg_detection_risk = [0] * original_grid.scene_length
        new_grid.avg_tracking_risk = [0] * original_grid.scene_length
        new_grid.avg_occ = [0] * original_grid.scene_length
        new_grid.total_obj = [0] * original_grid.scene_length
        new_grid.total_obj_sev = [0] * original_grid.scene_length
        new_grid.avg_occ_ranges = [[0] * len(original_grid.ranges) for _ in range(original_grid.scene_length)]
        
        new_grid.create_non_empty_grid()

        return new_grid

    def load_map_show_plots(dataroot, datafile_name, map_name, map_width, map_height, map_short, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY, risk_weights):
        # Entry point for the main simulation function
        print("Starting main function...")

        # Initialize the map object with given parameters
        map_const = Map(dataroot, map_name, map_width, map_height, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY)
        map_var = Map(dataroot, map_name, map_width, map_height, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY)
        maps = [map_const, map_var]

        # Create a folder structure to save the run results and plots
        run_folder_cons = os.path.join("Runs", map_short, f"scene {id} res={RESOLUTION}", 'Constant Power')
        run_folder_var = os.path.join("Runs", map_short, f"scene {id} res={RESOLUTION}", 'Variable Power')
        run_folders = [run_folder_cons, run_folder_var]

        # create a folder where all the comparison plots are made
        comparison_folder = os.path.join("Runs", map_short, f"scene {id} res={RESOLUTION}")
        os.makedirs(comparison_folder, exist_ok=True)

        power_profile_folder = os.path.join("Runs", map_short, f"scene {id} res={RESOLUTION}", 'Power Profiles')
        os.makedirs(power_profile_folder, exist_ok=True)

        plots_folders = []
        gif_folders = []
        scene_data_paths = []
        plots_folders = []
        
        layer_plot_paths = []
        pointclouds_folders = []
        pointclouds_overlay_folders = []

        risk_plots_folders = []
        occ_folders = []
        occ_hist_folders = []

        for run, run_folder in enumerate(run_folders):
            os.makedirs(run_folder, exist_ok=True)  # Ensure the directory exists

            # Create subdirectories for storing plots and GIFs
            plots_folders.append(os.path.join(run_folder, 'plots'))
            os.makedirs(plots_folders[run], exist_ok=True)

            gif_folders.append(os.path.join(run_folder, 'GIFs'))
            os.makedirs(gif_folders[run], exist_ok=True)

            # Paths for data and specific plots
            scene_data_paths.append(os.path.join(run_folder, datafile_name))
            layer_plot_paths.append(os.path.join(plots_folders[run], "layers.png"))

            # Subdirectories for specific plot types
            risk_plots_folders.append(os.path.join(plots_folders[run], "risks"))
            pointclouds_folders.append(os.path.join(plots_folders[run], "pointclouds"))
            pointclouds_overlay_folders.append(os.path.join(plots_folders[run], "pointclouds overlay"))
            occ_folders.append(os.path.join(plots_folders[run], "occurrence"))
            occ_hist_folders.append(os.path.join(plots_folders[run], "occurrence histograms"))

            # Ensure subdirectories exist
            os.makedirs(risk_plots_folders[run], exist_ok=True)
            os.makedirs(pointclouds_folders[run], exist_ok=True)
            os.makedirs(pointclouds_overlay_folders[run], exist_ok=True)
            os.makedirs(occ_folders[run], exist_ok=True)
            os.makedirs(occ_hist_folders[run], exist_ok=True)

        risk = Risk(risk_weights)

        # assigns layers for both simulation, the map var that calls it does not matter
        Map.assign_layers(scene_data_paths, maps, prnt=False)
        
        maps[0].grid.update_ETA(rang=LIDAR_RANGE, ego=maps[0].ego_positions, i=0)
        maps[1].grid.update_ETA(rang=LIDAR_RANGE, ego=maps[1].ego_positions, i=0)

        print('Normalising risks')
        risk.Normalise_and_calc_risks(maps, i=0)

        unchanged_risk_matrix = maps[0].grid.get_unchanged_static_risk_matrix()
        ETA_matrix = maps[0].grid.get_eta_weight_matrix()
        static_risk_matrix = maps[0].grid.get_static_risk_matrix()

        #static_risk_matrix = np.array(unchanged_risk_matrix) * np.array(ETA_matrix)

        Visualise.show_risk(unchanged_risk_matrix, 'Static Risk Map')
        Visualise.show_risk(ETA_matrix, 'ETA weight Map')
        Visualise.show_risk(static_risk_matrix, 'ETA masked Static Risk Map')