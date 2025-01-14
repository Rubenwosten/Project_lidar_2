import numpy as np

class ETA:
    def __init__(self, map):
        self.map = map
        self.sample = None
        self.sampleindex = None
        self.oud = None
        self.ego = map.ego_positions

    def update(self, sample, sample_index):
        self.sample = sample
        self.sampleindex = sample_index
        if self.sample!=self.oud:
            if self.sampleindex == 0:
                v = np.sqrt((self.ego[1][0]-self.ego[0][0])**2+(self.ego[1][1]-self.ego[0][1])**2)/0.5
            else:
                v = np.sqrt((self.ego[self.sampleindex][0]-self.ego[self.sampleindex-1][0])**2+(self.ego[self.sampleindex][1]-self.ego[self.sampleindex-1][1])**2)/0.5
            distance_eta_3 = 3*v
            for row in self.map.grid.grid:
            # Iterate through each cell in the current row
                for cell in row:
                    if cell.layer != 'empty':
                        d = ((self.ego[self.sampleindex][0]-cell.x)**2+(self.ego[self.sampleindex][1]-cell.y)**2)
                        if d >= distance_eta_3**2:
                            risk_eta = 0.5
                        else:
                            d = np.sqrt(d)
                            eta = d*v
                            risk_eta = 0.0667*eta**3-0.3*eta**2+0.0333*eta+1
                        cell.detect_risk[self.sampleindex] *= risk_eta
                        cell.track_risk[self.sampleindex] *= risk_eta
                        cell.static_risk[self.sampleindex] *= risk_eta
                        

            
