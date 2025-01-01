
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

map_name = 'boston-seaport'  #'singapore-onenorth'
map_short = 'Boston'

map_width = 2979.5
map_height = 2118.1

amount_cones = 8
max_power = 100
LIDAR_RANGE = 100 # 100 meter
OCC_ACCUM = 1 / 8 # full accumulation in 8 samples = 4 sec 
LIDAR_DECAY = 1 # amount of occurrence that goes down per lidar point

risk_weights = (1, 1, 1) # (0.5, 2, 10) # static, detection, tracking

scene_id = 4
RESOLUTION = 0.5 # meter
run_detect = True
run_obj = False
plot_layers = False
plot_pointcloud = False
show_pointcloud = False
plot_occ_hist = False

constant_power = True

def main(map_short, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY, constant_power):

    print("Starting main function...")
    map = Map(dataroot, map_name, map_width, map_height, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY)

    if constant_power:
        sim_type = 'Constant Power'
    else:
        sim_type = 'Variable Power'
    # Create a folder to save the run and plots if it doesn't already exist
    # Create the Run/Boston/scene 1 folder structure
    run_folder = os.path.join("Runs", map_short, f"scene {id} res={RESOLUTION}", sim_type)
    os.makedirs(run_folder, exist_ok=True)

    plots_folder = os.path.join(run_folder,'plots')
    os.makedirs(plots_folder, exist_ok=True)

    gif_folder = os.path.join(run_folder,'GIFs')
    os.makedirs(gif_folder, exist_ok=True)

    # Paths for data, plots, and subfolders
    scene_data_path = os.path.join(run_folder, "data")
    layer_plot_path = os.path.join(plots_folder, "layers.png")
    risk_plots_folder = os.path.join(plots_folder, "risks")
    pointclouds_folder = os.path.join(plots_folder, "pointclouds")
    pointclouds_overlay_folder = os.path.join(plots_folder, "pointclouds overlay")
    occ_folder = os.path.join(plots_folder, "occurrence")
    occ_hist_folder = os.path.join(plots_folder, "occurrence histograms")
    
    # Create subfolders
    os.makedirs(risk_plots_folder, exist_ok=True)
    os.makedirs(pointclouds_folder, exist_ok=True)
    os.makedirs(pointclouds_overlay_folder, exist_ok=True)
    os.makedirs(occ_folder, exist_ok=True)
    os.makedirs(occ_hist_folder, exist_ok=True)

    # Assign layers to the grid in parallel
    map.assign_layer(scene_data_path, prnt=False)

    map.save_grid(scene_data_path)
    
    # Generate and save the layer plot
    if plot_layers:
        Visualise.plot_layers(map.grid, layer_plot_path)

    # Initialize risk calculation
    risk = Risk()
    obj = Object(map)
    dec = Detect(map)
    powe = power(map,RESOLUTION, amount_cones,LIDAR_RANGE, max_power)
    sub = subsample(dataroot, map, amount_cones, LIDAR_RANGE)
    filt = object_filter(map)

    # Calculate risk for each sample
    for i, sample in enumerate(map.samples):
        #TODO add a check if it has already been set
        if run_obj:
            map.grid.total_obj[i], map.grid.total_obj_sev[i] = obj.update(sample=sample,x=0,y=0,sample_index=i, prnt=False)

        #TODO add a check if it has already been set
        if run_detect:
            dec.update(sample=sample, sample_index=i, prnt=False)
        
        # Save individual pointcloud plots
        if plot_pointcloud:
            Visualise.save_pointcloud_scatterplot(map, dec.lidarpoint, i, pointclouds_folder, overlay=False)
            Visualise.save_pointcloud_scatterplot(map, dec.lidarpoint, i, pointclouds_overlay_folder, overlay=True)
        
        print(f"sample {i} complete\n")

    # normalise the risk elements between 0 and 1
    # calculate the total risk based on the weights
    normalise_and_calc_risks(map, risk_weights)

    maxs = get_global_max(map=map)
    for i, sample in enumerate(map.samples):
        # update total variables
        map.update(sample=sample,i=i, weights=risk_weights)

        powe.sample = (sample, i)
        power_opti = powe.p_optimal
        print(power_opti)
        sub.sample = (sample, i, scene_id, power_opti)
        lidar_new = sub.subsamp
        count_new = sub.count_new
        print(count_new)
        print(sub.count)
        filt.update(sample, i, lidar_new, count_new)
        objs_scan = filt.object_scanned
        print(len(objs_scan))
        print(filt.count)
        
        # Save individual risk plots
        Visualise.plot_risks_maximised(map.grid, i, maxs, risk_plots_folder)
        Visualise.plot_occ(map.grid, i, occ_folder)   
        
        # plot occurrence range histograms
        if plot_occ_hist:
            Visualise.plot_occ_histogram(map, i, occ_hist_folder)

    # save the grid with the new risk values 
    map.save_grid(scene_data_path)

    Visualise.plot_avg_risks(map.grid, plots_folder)
    Visualise.plot_total_var(map.grid.avg_occ, 'Average Occurrence', plots_folder)
    Visualise.plot_total_var(map.grid.total_obj, 'Total Objects', plots_folder)
    Visualise.plot_total_var(map.grid.total_obj_sev, 'Total Object severity', plots_folder)
    Visualise.plot_avg_occ_histogram(map, plots_folder)

    # create gifs of all results
    Visualise.create_gif_from_folder(risk_plots_folder, os.path.join(gif_folder,'risks.gif'))
    Visualise.create_gif_from_folder(pointclouds_folder, os.path.join(gif_folder,'pointcloud.gif'))
    Visualise.create_gif_from_folder(pointclouds_overlay_folder, os.path.join(gif_folder,'pointcloud_layers.gif'))
    Visualise.create_gif_from_folder(occ_folder, os.path.join(gif_folder,'occurrence.gif'))
    Visualise.create_gif_from_folder(occ_hist_folder, os.path.join(gif_folder,'occurrence_hist.gif'))

    # check scene data size, if more than 100MB give a warning message to add it to the gitignore
    data_size = os.path.getsize(scene_data_path)
    if (data_size > 100000000):
        print(f'\nDATA FILE {scene_data_path} \nIS TOO BIG FOR GITHUB: ADD IT TO THE GITIGNORE FILE')
    print('Done')

def get_global_max(map):
    max_total = max(np.max(np.array(matrix)) for matrix in [map.grid.get_total_risk_matrix(i) for i in range(map.grid.scene_length)])
    max_static = np.max(np.array(map.grid.get_static_risk_matrix()))
    max_detect = max(np.max(np.array(matrix)) for matrix in [map.grid.get_detect_risk_matrix(i) for i in range(map.grid.scene_length)])
    max_track = max(np.max(np.array(matrix)) for matrix in [map.grid.get_track_risk_matrix(i) for i in range(map.grid.scene_length)])
    return (max_total, max_static, max_detect, max_track)

def normalise_and_calc_risks(map, weights):
    maxs = get_global_max(map)
    max_total, max_static, max_detect, max_track = [value if value > 0 else 1 for value in maxs]
    w_s, w_d, w_t = weights


    print(f'maxs before norm = {maxs}')
    print(f'max_track before norm = {max_track}')
    for row in map.grid.grid:
        for cell in row:
            cell.static_risk = cell.static_risk/max_static
            cell.detect_risk = [detect_risk/max_detect for detect_risk in cell.detect_risk]
            cell.track_risk = [track_risk/max_track for track_risk in cell.track_risk]
            for i in range(len(cell.detect_risk)):
                cell.total_risk[i] = w_s * cell.static_risk + w_d * cell.detect_risk[i] + w_t * cell.track_risk[i]
    
    maxs = get_global_max(map)
    max_total, max_static, max_detect, max_track = [value if value > 0 else 1 for value in maxs]
    print(f'maxs = after norm {maxs}')
    print(f'max_track after norm = {max_track}')

# This ensures that the code is only executed when the script is run directly
if __name__ == '__main__':
    print("Running as main module...")  # Debugging line
    start_time = time.time()
    main(map_short=map_short, 
        id=scene_id, 
        LIDAR_RANGE=LIDAR_RANGE, 
        RESOLUTION=RESOLUTION, 
        OCC_ACCUM=OCC_ACCUM, 
        LIDAR_DECAY=LIDAR_DECAY, 
        constant_power=constant_power)

    run_time = time.time() - start_time
    print(f'\nRunning took {timedelta(seconds=run_time)}')
