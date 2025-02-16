
# This is the file that executes all the code

import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import timedelta
import numpy as np
import re
import Cell
import os
from Grid import Grid 
from Map import Map
from Visualise import Visualise
from Risk import Risk
from Object import Object
from Dectetion import Detect
from Power import power
from Subsample import subsample
from Object_filter import object_filter
from Error import Error

import warnings
from shapely.errors import ShapelyDeprecationWarning

# Suppress ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

dataroot = r"C:/Users/Ruben/OneDrive/Bureaublad/data/sets/nuscenes"
#dataroot = r"C:/Users/marni/OneDrive/Documents/BEP 2024/data/sets/nuscenes"
#dataroot = r'C:/Users/Chris/Python scripts/BEP VALDERS/data/sets/nuscenes'

map_name = ['singapore-onenorth','boston-seaport','boston-seaport','boston-seaport','boston-seaport', 'singapore-queenstown','singapore-queenstown','singapore-hollandvillage','singapore-hollandvillage','singapore-hollandvillage'] #'singapore-onenorth'
map_short = ['singapore','Boston','Boston','Boston','Boston','singapore','singapore','singapore','singapore','singapore']
datafile_name = 'reinitialized_data.pkl'

map_width = 2979.5
map_height = 2118.1

amount_cones = 8
max_power = 12 # watt
procent = 0.5
LIDAR_RANGE = 100 # 100 meter
OCC_ACCUM = 1 / 8 # full accumulation in 8 samples = 4 sec 
LIDAR_DECAY = 1 # amount of occurrence that goes down per lidar point
probability_threshold = 0.975 

risk_weights = (1, 4, 2) # (0.5, 2, 10) # static, detection, tracking

scene_id = 1
RESOLUTION = 0.5 # meter

run_detect = True
run_obj = True
run_power = True

plot_layers = True
plot_pointcloud = True
show_pointcloud = False
plot_occ_hist = True
plot_occ = True
plot_risk = True
plot_power_profile = True


def main(map_name, map_short, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY):
    # Entry point for the main simulation function
    print("Starting main function...")

    # Initialize the map object with given parameters
    map_const = Map(dataroot, map_name, map_width, map_height, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY)
    map_var = Map(dataroot, map_name, map_width, map_height, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY)
    maps = [map_const, map_var]

    # Create a folder structure to save the run results and plots
    scene_name = os.path.join("Runs", map_short, f"scene {id} res={RESOLUTION} Power = {100*procent}%")
    run_folder_cons = os.path.join(scene_name,'Constant Power')
    run_folder_var = os.path.join(scene_name, 'Variable Power')
    run_folders = [run_folder_cons, run_folder_var]

    # create a folder where all the comparison plots are made
    comparison_folder = os.path.join(scene_name)
    os.makedirs(comparison_folder, exist_ok=True)

    power_profile_folder = os.path.join(scene_name, 'Power Profiles')
    os.makedirs(power_profile_folder, exist_ok=True)

    plots_folders = []
    gif_folders = []
    scene_data_paths = []
    plots_folders = []
    
    layer_plot_paths = []
    pointclouds_folders = []
    pointclouds_overlay_folders = []
    pointclouds_overlay_removed_folders = []

    risk_plots_folders = []
    expected_risk_plots_folders = []
    prob_plots_folders = []
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
        expected_risk_plots_folders.append(os.path.join(plots_folders[run], "expected risks"))
        prob_plots_folders.append(os.path.join(plots_folders[run], "probability"))
        pointclouds_folders.append(os.path.join(plots_folders[run], "pointclouds"))
        pointclouds_overlay_folders.append(os.path.join(plots_folders[run], "pointclouds overlay"))
        pointclouds_overlay_removed_folders.append(os.path.join(plots_folders[run], "removed points overlay"))
        occ_folders.append(os.path.join(plots_folders[run], "occurrence"))
        occ_hist_folders.append(os.path.join(plots_folders[run], "occurrence histograms"))

        # Ensure subdirectories exist
        os.makedirs(risk_plots_folders[run], exist_ok=True)
        os.makedirs(expected_risk_plots_folders[run], exist_ok=True)
        os.makedirs(prob_plots_folders[run], exist_ok=True)
        os.makedirs(pointclouds_folders[run], exist_ok=True)
        os.makedirs(pointclouds_overlay_folders[run], exist_ok=True)
        os.makedirs(pointclouds_overlay_removed_folders[run], exist_ok=True)
        os.makedirs(occ_folders[run], exist_ok=True)
        os.makedirs(occ_hist_folders[run], exist_ok=True)

    # assigns layers for both simulation, the map var that calls it does not matter
    Map.assign_layers(scene_data_paths, maps, prnt=False)

    # Optionally generate and save a plot of the map layers
    if plot_layers:
        Visualise.plot_layers(maps[0].grid, layer_plot_paths[0])
        Visualise.plot_layers(maps[1].grid, layer_plot_paths[1])

    # Initialize risk calculation
    powe1 = power(maps[0], amount_cones, max_power,procent, subsample(maps[0], amount_cones, probability_threshold), object_filter(maps[0]), constant_power = True)
    powe2 = power(maps[1], amount_cones, max_power,procent, subsample(maps[1], amount_cones, probability_threshold), object_filter(maps[1]),  constant_power=False)

    # Initialize components for risk calculation, object tracking, and detection
    risk = Risk(risk_weights)
    objs = [Object(maps[0], constant_power=True), Object(maps[1], constant_power=False)]
    decs = [Detect(maps[0], constant_power=True), Detect(maps[1], constant_power=False)]
    
    # Process each sample in the map
    for i, sample in enumerate(maps[0].samples):
        # Update object data if required
        if not i == 0:
            if run_obj:
                print('Updating objects')
                maps[0].grid.total_obj[i], maps[0].grid.total_obj_sev[i] = objs[0].update(sample=sample, x=0, y=0, sample_index=i ,object_list_new=objs_scan_const)
                maps[1].grid.total_obj[i], maps[1].grid.total_obj_sev[i] = objs[1].update(sample=sample, x=0, y=0, sample_index=i ,object_list_new=objs_scan_var)

            # Update detection data if required
            if run_detect:
                print('Updating detection')
                decs[0].update(sample=sample, sample_index=i, lidar_new=lidar_new_const)
                decs[1].update(sample=sample, sample_index=i, lidar_new=lidar_new_var)
        
        # calculate ETA weights
        print('Calculating ETA weights')
        maps[0].grid.update_ETA(rang=LIDAR_RANGE, ego=maps[0].ego_positions, i=i)
        maps[1].grid.update_ETA(rang=LIDAR_RANGE, ego=maps[1].ego_positions, i=i)

        print('Normalising risks')
        risk.Normalise_and_calc_risks(maps, i)
        
        if run_power:
            print('Updating power profile')
            # update the power profile for the next sample
            lidar_new_const, objs_scan_const, lidar_removed_cons = powe1.update(curr_sample=sample, curr_sample_index=i, scene_id=scene_id)
            lidar_new_var, objs_scan_var, lidar_removed_var = powe2.update(curr_sample=sample, curr_sample_index=i, scene_id=scene_id)
        
        print('Calculating average variables')
        maps[0].grid.update_avg_vars(ego=maps[0].ego_positions, i=i, weights=risk_weights)
        maps[1].grid.update_avg_vars(ego=maps[1].ego_positions, i=i, weights=risk_weights)
            
        # Save point cloud plots for each sample
        if plot_pointcloud:
            Visualise.save_pointcloud_scatterplot(maps[0], lidar_new_const, i, pointclouds_folders[0], overlay=False)
            Visualise.save_pointcloud_scatterplot(maps[0], lidar_new_const, i, pointclouds_overlay_folders[0], overlay=True)
            Visualise.save_pointcloud_scatterplot(maps[0], lidar_removed_cons, i, pointclouds_overlay_removed_folders[0], overlay=True, title='Removed pointcloud')
            Visualise.save_pointcloud_scatterplot(maps[1], lidar_new_var, i, pointclouds_folders[1], overlay=False)
            Visualise.save_pointcloud_scatterplot(maps[1], lidar_new_var, i, pointclouds_overlay_folders[1], overlay=True)
            Visualise.save_pointcloud_scatterplot(maps[1], lidar_removed_var, i, pointclouds_overlay_removed_folders[1], overlay=True, title='Removed pointcloud')


        if plot_risk:
            Visualise.plot_risks(maps[0].grid, i, risk_plots_folders[0])
            Visualise.plot_risks(maps[1].grid, i, risk_plots_folders[1])
            Visualise.plot_prob(maps[0].grid, i, prob_plots_folders[0])
            Visualise.plot_prob(maps[1].grid, i, prob_plots_folders[1])
            Visualise.plot_expected_risk(maps[0].grid, i, expected_risk_plots_folders[0])
            Visualise.plot_expected_risk(maps[1].grid, i, expected_risk_plots_folders[1])

        if plot_occ:
            Visualise.plot_occ(maps[0].grid, i, occ_folders[0])
            Visualise.plot_occ(maps[1].grid, i, occ_folders[1])

        if plot_power_profile:
            Visualise.plot_power_profile(powe1.p_optimal, powe2.p_optimal, i, power_profile_folder)

        if plot_occ_hist:
            Visualise.plot_occ_histogram(maps[0], i, occ_hist_folders[0])
            Visualise.plot_occ_histogram(maps[1], i, occ_hist_folders[1])

        print(f"sample {i} complete\n")

    # Save updated map grid with new risk values
    maps[0].save_grid(scene_data_paths[0])
    maps[1].save_grid(scene_data_paths[1])

    Visualise.plot_avg_risks(maps, comparison_folder)
    Visualise.plot_avg_occ(maps, comparison_folder)
    Visualise.plot_total_var(maps[0].grid.total_obj, maps[1].grid.total_obj, 'Total Objects', comparison_folder)
    Visualise.plot_total_var(maps[0].grid.total_obj_sev, maps[1].grid.total_obj_sev, 'Total Object severity', comparison_folder)
    Visualise.plot_total_var(powe1.t_cost, powe2.t_cost, 'Cost function', comparison_folder)
    Visualise.plot_removed_pointcount(powe1.sub.verschil, powe2.sub.verschil, comparison_folder)
    Visualise.plot_avg_occ_histogram(maps, comparison_folder)

    error = Error(maps)
    error.save_results_to_file(os.path.join(comparison_folder, 'error_results.txt'))

    Visualise.create_gif_from_folder(power_profile_folder, os.path.join(power_profile_folder, 'power_profile.gif'))
    # Generate summary plots for the simulation
    for run, map in enumerate(maps):
        # Create GIFs for visualizing results
        gif_folder = gif_folders[run]
        Visualise.create_gif_from_folder(risk_plots_folders[run], os.path.join(gif_folder, 'risks.gif'))
        Visualise.create_gif_from_folder(expected_risk_plots_folders[run], os.path.join(gif_folder, 'expected_risks.gif'))
        Visualise.create_gif_from_folder(pointclouds_folders[run], os.path.join(gif_folder, 'pointcloud.gif'))
        Visualise.create_gif_from_folder(pointclouds_overlay_folders[run], os.path.join(gif_folder, 'pointcloud_layers.gif'))
        Visualise.create_gif_from_folder(occ_folders[run], os.path.join(gif_folder, 'occurrence.gif'))
        Visualise.create_gif_from_folder(occ_hist_folders[run], os.path.join(gif_folder, 'occurrence_hist.gif'))

    for run, scene_data_path in enumerate(scene_data_paths):
        # Check the size of the scene data file and warn if it exceeds 100MB
        data_size = os.path.getsize(scene_data_path)
        if data_size > 100000000:  # File size threshold in bytes
            scene_data_path = scene_data_path.replace('\\', '/')  # Ensure path is in a standard format
            try:
                # Read .gitignore contents
                with open('.gitignore', 'r') as f:
                    gitignore_contents = f.read().splitlines()

                # Add the path to .gitignore if it's not already listed
                if not any(scene_data_path in line for line in gitignore_contents):
                    with open('.gitignore', 'a') as f:  # Append mode
                        f.write(f'\n{scene_data_path}\n')
                    print(f'\nDATA FILE {scene_data_path} \nIS TOO BIG FOR GITHUB: ADDED TO THE GITIGNORE FILE')
                else:
                    print(f'\nDATA FILE {scene_data_path} \nIS TOO BIG FOR GITHUB: ALREADY IN THE GITIGNORE FILE')
            except Exception as e:
                print(f'Error handling .gitignore file: {e}')



    print('Done')  # End of the main function


# This ensures that the code is only executed when the script is run directly
if __name__ == '__main__':
    print("Running as main module...")  # Debugging line
    start_time = time.time()
    main(map_name=map_name[0],map_short=map_short[0], 
    id=0, 
    LIDAR_RANGE=LIDAR_RANGE, 
    RESOLUTION=RESOLUTION, 
    OCC_ACCUM=OCC_ACCUM, 
    LIDAR_DECAY=LIDAR_DECAY)

    run_time = time.time() - start_time
    print(f'\nRunning took {timedelta(seconds=run_time)}')

