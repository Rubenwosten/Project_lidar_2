
# This class handles the risk function(s)
class Risk:

    def __init__(self, weights):
        if len(weights) != 3:
            raise ValueError("Weights must be a tuple of length 3 (w_static, w_detect, w_track).")
        self.weights = weights
        

    def CalcRisk(self, map, i):
        """
        Calculate the total risk as a weighted sum of static_risk, detect_risk, and track_risk.

        :param weights: Tuple of three weights (w_static, w_detect, w_track)
        :return: Total calculated risk
        """
        w_static, w_detect, w_track = self.weights

        for cells in map.grid.grid:
            for cell in cells:
                cell.total_risk[i] = w_static * cell.static_risk + w_detect * cell.detect_risk[i] + w_track * cell.track_risk[i]
        
    def normalise_and_calc_risks(self, map, maxs):
        """
        Normalizes risks and calculates total risk per cell using given weights.
        """
        max_total, max_static, max_detect, max_track = [value if value > 0 else 1 for value in maxs]
        w_s, w_d, w_t = self.weights
        
        for row in map.grid.grid:
            for cell in row:
                cell.static_risk /= max_static
                cell.detect_risk = [detect_risk/max_detect for detect_risk in cell.detect_risk]
                cell.track_risk = [track_risk/max_track for track_risk in cell.track_risk]
                for i in range(len(cell.detect_risk)):
                    cell.total_risk[i] = w_s * cell.static_risk + w_d * cell.detect_risk[i] + w_t * cell.track_risk[i]/max_total
