
# This class creates a grid 
from Cell import Cell
import numpy as np
import math

class Grid:

    def __init__(self, patch, resolution, scene_length, RANGE, prnt=False):
        # grid vars
        self.patch = patch 
        x_min, x_max , y_min, y_max = patch
        self.res = resolution
        self.scene_length = scene_length
        self.width = int((x_max - x_min)/resolution)
        self.length = int((y_max - y_min)/resolution)
        self.xarray = np.linspace(x_min, x_max, self.width)
        self.yarray = np.linspace(y_min, y_max, self.length)

        # Total vars
        self.cells_off_interest = []
        self.avg_total_risk = [0] * scene_length
        self.avg_static_risk = [0] * scene_length
        self.avg_detection_risk = [0] * scene_length
        self.avg_tracking_risk = [0] * scene_length
        self.avg_occ = [0] * scene_length
        self.total_obj = [0] * scene_length
        self.total_obj_sev = [0] * scene_length
        
        # Initialize total_occ_ranges as a 2D list
        self.ranges = np.linspace(RANGE/10, RANGE, 10)  # Default ranges: 0-10, 10-20, ..., 90-100
        self.avg_occ_ranges = [[0] * (len(self.ranges)) for _ in range(scene_length)]

        # grid instatiation
        self.grid = [[Cell(self.xarray[x], self.yarray[y], scene_length) for y in range(self.length)] for x in range(self.width)]

        self.has_assigned_layers = False
        if prnt:
            print('grid of width {} and length {} was created with {} elements'.format(self.width, self.length, self.width * self.length))
        
    def get_cell(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.length:
            return self.grid[x][y]
        else:
            raise IndexError(f"Cell coordinates ({x}, {y}) are out of bounds. "
                             f"Grid size is width={self.width}, length={self.length}.")
    
    def count_layers(self):
        """
        Count the occurrences of each layer type in the grid.

        :return: Dictionary with layer names as keys and their counts as values
        """
        layer_counts = {layer: 0 for layer in Cell.priority_layers}

        for row in self.grid:
            for cell in row:
                for layer_name in cell.layers:
                    if layer_name in layer_counts:
                        layer_counts[layer_name] += 1

        return layer_counts
    
    def create_non_empty_grid(self):
        self.non_empty_grid = []
        for cells in self.grid:
            for cell in cells:
                if cell.layer != 'empty':
                    self.non_empty_grid.append(cell)

    def update_ETA(self, rang, ego, i):
        self.cells_off_interest = self.circle_of_interrest(rang, ego[i])

        self.cells_off_interest = [cell for cell in self.cells_off_interest if cell.layer != 'empty']

        self.ETA_calcs(i, ego)


    def update_avg_vars(self, ego, i, weights):
        
        num_nonempty_cells = len(self.cells_off_interest)
        self.calc_avg_vars(num_nonempty_cells, i, weights, ego[i])

        
    def ETA_calcs(self, i, ego):
        # Calculate velocity
        v = self.calc_v(i, ego)
        distance_eta_3 = 3 * v

        # Separate cells into those inside and outside the circle of interest
        cells_in_circle = self.cells_off_interest
        cells_outside_circle = [cell for row in self.grid for cell in row if cell not in cells_in_circle and cell.layer != 'empty']

        # Handle cells outside the circle of interest: Directly assign risk_eta = 0.5
        for cell in cells_outside_circle:
            cell.ETA_weight = 0.5
            cell.detect_risk[i] *= 0.5
            cell.track_risk[i] *= 0.5
            cell.static_risk = cell.unchanged_static_risk * 0.5

        # Handle cells inside the circle of interest
        for cell in cells_in_circle:
            d = np.sqrt((ego[i][0] - cell.x) ** 2 + (ego[i][1] - cell.y) ** 2)
            if d >= distance_eta_3:
                cell.ETA_weight = 0.5
            else:
                eta = d / v
                cell.ETA_weight = 0.0667 * eta**3 - 0.3 * eta**2 + 0.0333 * eta + 1
            cell.detect_risk[i] *= cell.ETA_weight
            cell.track_risk[i] *= cell.ETA_weight
            cell.static_risk = cell.unchanged_static_risk * cell.ETA_weight
    
    def calc_v(self, i, ego):
        if i == 0:
            v = np.sqrt((ego[1][0]-ego[0][0])**2+(ego[1][1]-ego[0][1])**2)/0.5
        else:
            v = np.sqrt((ego[i][0]-ego[i-1][0])**2+(ego[i][1]-ego[i-1][1])**2)/0.5
        return v
    
    def calc_avg_vars(self, num_nonempty_cells, i, weights, ego):
        # reset the variables 
        self.avg_total_risk[i] = 0
        self.avg_static_risk[i] = 0
        self.avg_detection_risk[i] = 0
        self.avg_tracking_risk[i] = 0
        self.avg_occ[i] = 0

        for cell in self.cells_off_interest:            
            # cell variables
            self.avg_static_risk[i] += cell.static_risk
            self.avg_detection_risk[i] += cell.detect_risk[i]
            self.avg_tracking_risk[i] += cell.track_risk[i]
            self.avg_total_risk[i] += cell.total_risk[i]
            self.avg_occ[i] += cell.occ[i]

        self.avg_occ[i] /= num_nonempty_cells

        # Initialize sets for processed cells
        smaller_range_cells = set()

        # Loop through the ranges to calculate the total occupancy for each range
        for idx, current_range in enumerate(self.ranges):

            # Get cells for the current range as a set
            current_range_cells = set(self.circle_of_interrest(current_range, ego))

            # Subtract the cells in the previous range (smaller range) from the current range
            exclusive_cells_in_range = current_range_cells - smaller_range_cells

            # Sum the occupancy values for the exclusive cells in this range
            exclusive_cells_in_range = [cell for cell in exclusive_cells_in_range if cell.layer != 'empty']
            count_non_empty_cells = len(exclusive_cells_in_range)
            self.avg_occ_ranges[i][idx] = sum(cell.occ[i] for cell in exclusive_cells_in_range)/count_non_empty_cells

            #print(f'current_range = {current_range}\t count_non_empty_cells = {count_non_empty_cells}\t self.total_occ_ranges[{i}][{idx}] = {round(self.total_occ_ranges[i][idx],4)}')

            # Update the smaller range cells to include the current range cells
            smaller_range_cells.update(current_range_cells)
        w_static, w_detect, w_track = weights

        self.avg_static_risk[i] /= num_nonempty_cells
        self.avg_detection_risk[i] /= num_nonempty_cells
        self.avg_tracking_risk[i] /= num_nonempty_cells
        self.avg_total_risk[i] = w_static * self.avg_static_risk[i] + w_detect * self.avg_detection_risk[i] + w_track * self.avg_tracking_risk[i]
        

    def circle_of_interrest(self, range, ego):
        if range == 0: 
            return []
        circle_interrest = []
        for cell in self.non_empty_grid:
            x= cell.x
            y= cell.y
            distance = (y-ego[1])**2 + (x-ego[0])**2
            # use quared distance to negate the computationally heavy sqrt function
            if distance < (range**2):
                circle_interrest.append(cell)
        return circle_interrest
    
    def get_layer_matrix(self):
        return [[cell.layer for cell in row] for row in self.grid]
    
    def get_eta_weight_matrix(self):
        return [[cell.ETA_weight for cell in row] for row in self.grid]
    
    def get_total_risk_matrix(self, i):
        return [[cell.total_risk[i] for cell in row] for row in self.grid]
    
    def get_static_risk_matrix(self):
        return [[cell.static_risk for cell in row] for row in self.grid]
    
    def get_detect_risk_matrix(self, i):
        return [[cell.detect_risk[i] for cell in row] for row in self.grid]
    
    def get_track_risk_matrix(self, i):
        return [[cell.track_risk[i] for cell in row] for row in self.grid]
    
    def get_occ_matrix(self, i):
        return [[cell.occ[i] for cell in row] for row in self.grid]

    def to_dict(self):
        """
        Convert the Grid object into a dictionary format for saving.
        """
        
        return {
            'patch': self.patch,
            'resolution': self.res,
            'scene length': self.scene_length,
            'range':self.ranges[-1],
            'width': self.width,
            'length': self.length,
            'grid': [[cell.to_dict() for cell in row] for row in self.grid],  # Convert all cells to dictionaries
            'has_assigned_layers': self.has_assigned_layers,
            'total total risk': self.avg_total_risk,
            'total static risk': self.avg_static_risk,
            'total detection risk': self.avg_detection_risk,
            'total tracking risk': self.avg_tracking_risk,
            'total occ': self.avg_occ,
            'total obj': self.total_obj,
            'total obj sev': self.total_obj_sev
        }


    @staticmethod
    def from_dict(grid_dict):
        """
        Convert a dictionary back into a Grid object.
        """
        # Extract original patch and resolution
        patch = grid_dict['patch']
        resolution = grid_dict['resolution']
        scene_length = grid_dict['scene length']
        # Recreate the Grid object with the exact same patch and resolution
        grid = Grid(patch=patch, resolution=resolution, scene_length=scene_length, RANGE=100)

        # Restore other attributes
        grid.width = grid_dict['width']
        grid.length = grid_dict['length']
        grid.has_assigned_layers = grid_dict['has_assigned_layers']
        # Rebuild the grid with Cell objects
        grid.grid = [
            [Cell.from_dict(cell_dict, scene_length) for cell_dict in row]
            for row in grid_dict['grid']
        ]
        grid.avg_total_risk = grid_dict['total total risk']
        grid.avg_static_risk = grid_dict['total static risk']
        grid.avg_detection_risk = grid_dict['total detection risk']
        grid.avg_tracking_risk = grid_dict['total tracking risk']
        grid.avg_occ = grid_dict['total occ']
        grid.total_obj = grid_dict['total obj']
        grid.total_obj_sev = grid_dict['total obj sev']
        
        return grid

        
