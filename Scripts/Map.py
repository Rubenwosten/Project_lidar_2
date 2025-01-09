import numpy as np
import time
from tqdm import tqdm
from Grid import Grid
import pickle
import os
import shutil

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap


class Map:

    def __init__(self, dataroot, map_name, map_width, map_height, scene_id, RANGE, RES, OCC_ACCUM, LIDAR_DECAY, prnt=False) -> None:
        self.dataroot = dataroot
        self.range = RANGE
        # get the correct nuscenes object
        self.nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)

        self.OCC_ACCUM = OCC_ACCUM
        self.LIDAR_DECAY = LIDAR_DECAY

        self.nusc_map = NuScenesMap(dataroot=dataroot, map_name=map_name)
        self.bitmap = BitMap(self.nusc_map.dataroot, self.nusc_map.map_name, 'basemap')

        # save the width and height of the map
        self.map_width = map_width
        self.map_height = map_height

        # get the specific scene based on the scene id 
        self.scene, first, last = self.get_scene(scene_id)

        # get the ego positions of the car for all samples of the scene
        self.samples,lidar_samples = self.samples_scene(first, last)
        self.ego_positions = self.ego_pos(lidar_samples)

        # make a patch based on the min and max of the coordinates plus the range of the lidar
        self.patch = self.get_patch(self.ego_positions, RANGE)
        if prnt:
            print(f'ego patch = {self.get_patch(self.ego_positions, 0)}')
            print('patch = {}'.format(self.patch))

        # initialise a cell grid 
        self.grid = Grid(self.patch, RES, len(self.samples), RANGE)

        # get all records within your map
        self.rec = self.get_records_in_patch(self.patch)

    def update(self, i, weights):
        self.grid.calc_avg_vars(rang=self.range, ego=self.ego_positions[i], i=i, weights=weights)

    def get_scene(self, index):
        #sample array
        self.scene = self.nusc.scene[index]
        first = self.scene['first_sample_token']
        last = self.scene['last_sample_token']
        return self.scene, first, last

    # This function loads in each sample in the the scene
    # returns a numpy array of the samples and a numpy array of the lidar sample
    def samples_scene(self, first,last):
        samples = np.empty(0)
        lidar_sample = np.empty(0)
        sample = first

        while sample != last:
            samples = np.append(samples, sample)
            info = self.nusc.get('sample', sample)
            sample = info['next']
            lidar_sample = np.append(lidar_sample, info['data']['LIDAR_TOP'])
        samples = np.append(samples, last)
        info_last= self.nusc.get('sample', last)
        lidar_sample = np.append(lidar_sample, info_last['data']['LIDAR_TOP'])
        return samples,lidar_sample


    # This function determines the path of the ego vehicle
    # returns a numpy array of the ego vehicles position
    def ego_pos(self, lidar_samples):
        ego_trans = np.empty((len(lidar_samples),3))
        i = 0
        for i in range(len(lidar_samples)):
            info = self.nusc.get('sample_data', lidar_samples[i])
            info_2 = self.nusc.get('ego_pose', info['ego_pose_token'])
            ego_trans[i] = info_2['translation']
            i += 1
        return ego_trans
        

    def get_patch(self, ego, range):
        x_min = np.min(ego[:,0]) - range
        x_max = np.max(ego[:,0]) + range
        y_min = np.min(ego[:,1]) - range
        y_max = np.max(ego[:,1]) + range
        return (x_min, x_max, y_min, y_max)


    def assign_lay(self, prnt = False):
        elements = self.grid.width * self.grid.length
        print(f"Assigning layers to the grid with {elements} elements.")

        start_time = time.time()
        for i, x in enumerate(tqdm(self.grid.xarray, desc='Assigning Layers')):
            if prnt:
                print(f"Assigning for i = {i} and x = {x} at time = {time.time() - start_time:.2f}")
            for j, y in enumerate(self.grid.yarray):
                self.grid.grid[i][j].layers = self.nusc_map.layers_on_point_v2(x, y, self.rec)
                self.grid.grid[i][j].assign_layer(prnt=False)
        self.has_assigned_layers = True

        elapsed_time = time.time() - start_time
        print(f"Grid layers were assigned in {elapsed_time} seconds")

    @staticmethod
    # This function assigns the layers variable of each cell based on the records within the map
    def assign_layers(filenames, maps, prnt=False):
        """
        Assigns the layers variable of each cell based on the records within the map.
        Adds the resolution value to the filename dynamically.

        :param filenames: A tuple containing the filenames for constant and variable power simulations
        :param prnt: Whether to print debug information
        """
        filename_cons, filename_var = filenames

        # Handle the constant power simulation
        if os.path.exists(filename_cons):
            print(f"File '{filename_cons}' was found. Loading ...")
            maps[0].load_grid(filename_cons)
        else:
            print(f"File '{filename_cons}' not found. Assigning layers to the grid.")
            maps[0].assign_lay(prnt)

            # Save the updated grid to the file
            maps[0].save_grid(filename_cons)
            print(f"Grid saved to '{filename_cons}'.")

        # Handle the variable power simulation
        if os.path.exists(filename_var):
            print(f"File '{filename_var}' was found. Loading ...")
            maps[1].load_grid(filename_var)
        else:
            print(f"File '{filename_var}' not found. Copying from constant power simulation.")
            shutil.copy(filename_cons, filename_var)
            print(f"File copied to '{filename_var}'.")
            maps[1].load_grid(filename_var)


    # this function getsall the records within the patch of the map
    def get_records_in_patch(self, patch):
        records_within_patch = self.nusc_map.get_records_in_patch(patch, self.nusc_map.non_geometric_layers, mode='intersect')
        rec = {}
        layer_names=['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']

        for layer in layer_names:
            rec[layer] = {}
            for record in records_within_patch[layer]:
                info = self.nusc_map.get(layer, record)
                rec[layer][record] = info
        return rec
    
    def get_global_max_timestep(self, i):
        max_total = np.max(np.array(self.grid.get_total_risk_matrix(i)))
        max_static = np.max(np.array(self.grid.get_static_risk_matrix()))
        max_detect = np.max(np.array(self.grid.get_detect_risk_matrix(i)))
        max_track = np.max(np.array(self.grid.get_track_risk_matrix(i)))
        return (max_total, max_static, max_detect, max_track)

    def get_global_max(self):
        """
        Finds the global maximum values for total, static, detect, and track risks in the map.

        Returns:
        tuple: Maximum values for (total, static, detect, track) risks.
        """
        max_total = max(np.max(np.array(matrix)) for matrix in [self.grid.get_total_risk_matrix(i) for i in range(self.grid.scene_length)])
        max_static = np.max(np.array(self.grid.get_static_risk_matrix()))
        max_detect = max(np.max(np.array(matrix)) for matrix in [self.grid.get_detect_risk_matrix(i) for i in range(self.grid.scene_length)])
        max_track = max(np.max(np.array(matrix)) for matrix in [self.grid.get_track_risk_matrix(i) for i in range(self.grid.scene_length)])
        return (max_total, max_static, max_detect, max_track)
    
    # Save the grid object
    def save_grid(self, filename):
        print('\nSAVING MAP: DO NOT KILL TERMINAL\n')
        with open(filename, 'wb') as f:
            pickle.dump(self.grid.to_dict(), f)  # Save the grid's dictionary representation
        print('Saved map: can now kill terminal')

    # Load the grid object
    def load_grid(self, filename):
        print('\nLOADING MAP: DO NOT KILL TERMINAL\n')
        with open(filename, 'rb') as f:
            grid_dict = pickle.load(f)  # Load the dictionary from the file
            self.grid = Grid.from_dict(grid_dict)
            return self.grid # Reconstruct the grid from the dictionary
        print('loaded map: can now kill terminal')

    
