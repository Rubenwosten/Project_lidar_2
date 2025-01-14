import numpy as np
# This class handles the risk function(s)
class Risk:

    def __init__(self, weights):
        if len(weights) != 3:
            raise ValueError("Weights must be a tuple of length 3 (w_static, w_detect, w_track).")
        self.weights = weights
        

    def Normalise_and_calc_risks_new(self, maps, i):
        """
        Calculate the total risk as a weighted sum of static_risk, detect_risk, and track_risk.

        :param weights: Tuple of three weights (w_static, w_detect, w_track)
        :return: Total calculated risk
        """

        # Retrieve global maxima for visualization scaling
        maxs_cons = maps[0].get_global_max_timestep(i)
        maxs_var = maps[1].get_global_max_timestep(i)

        # Calculate the biggest maxima across both simulations
        maxs = tuple(max(cons, var) for cons, var in zip(maxs_cons, maxs_var))
        max_total, max_static, max_detect, max_track = [value if value > 0 else 1 for value in maxs]
        print(f'maxs before norm = {(max_total, max_static, max_detect, max_track)}')
        w_s, w_d, w_t = self.weights

        for map in maps:
            for cells in map.grid.grid:
                for cell in cells:
                    cell.static_risk /= max_static
                    cell.detect_risk[i] /= max_detect
                    cell.track_risk[i] /= max_track
                    cell.total_risk[i] = w_s * cell.static_risk + w_d * cell.detect_risk[i] + w_t * cell.track_risk[i]
        
        max_total_cons = np.max(np.array(maps[0].grid.get_total_risk_matrix(i)))
        max_total_var = np.max(np.array(maps[1].grid.get_total_risk_matrix(i)))
        max_total = max(max_total_cons, max_total_var)
        for map in maps:
            for cells in map.grid.grid:
                for cell in cells:
                    cell.total_risk[i] /= max_total


    def normalise_and_calc_risks(self, maps):
        """
        Normalizes risks and calculates total risk per cell using given weights.
        """
        # Retrieve global maxima for visualization scaling
        maxs_cons = maps[0].get_global_max()
        maxs_var = maps[1].get_global_max()

        # Calculate the biggest maxima across both simulations
        maxs = tuple(max(cons, var) for cons, var in zip(maxs_cons, maxs_var))
        max_total, max_static, max_detect, max_track = [value if value > 0 else 1 for value in maxs]
        w_s, w_d, w_t = self.weights
        print(f"maxs before norm = {(max_total, max_static, max_detect, max_track)}")
        
        for map in maps:
            for row in map.grid.grid:
                for cell in row:
                    cell.static_risk /= max_static
                    cell.detect_risk = [detect_risk/max_detect for detect_risk in cell.detect_risk]
                    cell.track_risk = [track_risk/max_track for track_risk in cell.track_risk]
                    for i in range(len(cell.detect_risk)):
                        cell.total_risk[i] = w_s * cell.static_risk + w_d * cell.detect_risk[i] + w_t * cell.track_risk[i]/max_total
        
        # Retrieve global maxima for visualization scaling
        maxs_cons = maps[0].get_global_max()
        maxs_var = maps[1].get_global_max()

        # Calculate the biggest maxima across both simulations
        maxs = tuple(max(cons, var) for cons, var in zip(maxs_cons, maxs_var))
        max_total, max_static, max_detect, max_track = [value if value > 0 else 1 for value in maxs]
        print(f"maxs after norm = {(max_total, max_static, max_detect, max_track)}")
