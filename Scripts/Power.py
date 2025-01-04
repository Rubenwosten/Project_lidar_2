
import numpy as np
import math
from scipy.optimize import minimize
#lidar_parameters:
erx = 0.9 # receiver optics effeciency
etx = 0.9 # emmitter optics effeciency
n = 1 #target reflectivity
D = 25*pow(10,-3) #diameter lens 25 mm
Aovx = 1/np.pi #1 graden in radialen
Aovy = 1/np.pi #1 graden in radialen
phi_amb = 13.27 #W/m^2 gekozen via tabel want test wast delta labda = 50 nm
Nshots = 1
Ro = 0.9 #We kiezen een ADP lidar met een golflengte van 1550 nm
M = 30
F = 7
Bn = 1* 10**6 #Bandwidth 1 MHz
Id = 150*10**-9 # Dark current 150 nA
Rf = 10*10**3 #feed resistance 10 K ohm
vamp = 28*10**-9 # amplifier input voltage noice density 28 nV/sqrt(Hz)
T =293 #20 graden celsius in K
Kb = 1.380649*10**-23 # boltzmann constant
e = 1.602*10**-19 # elementaire lading
P_false = 10**-4 # false trigger mochten klein zetten
freq = 16.6667
T =1/freq


class power:
    def __init__(self, map, n, max_power, sub, filt):
        self.map = map
        self.reso = map.grid.res
        self.n_cones = n
        self.max_range = map.range
        self.ego = self.map.ego_positions
        self._sampleindex = None
        self._sample = None
        self.p_max= max_power
        self.oud = None
        self.sub = sub
        self.filt = filt

            

    def update(self, sample, sample_index, scene_id):
        self._sample = sample 
        self._sampleindex = sample_index
        cells = self.map.grid.circle_of_interrest(self.max_range,self.ego[self._sampleindex])
        cones = self.assign_cell_to_cone(cells)
        total_risk = 0
        total_risk_per_cone = np.zeros(self.n_cones)
        for cone_id, cone_cells in cones.items():
            for cell,distance in cone_cells:
                risk_cell = cell.total_risk[self._sampleindex]

                total_risk+=risk_cell
                total_risk_per_cone[cone_id] +=risk_cell
        p = np.zeros(self.n_cones)
        p_intial = np.zeros(self.n_cones)
        for cone, cone_cells in cones.items():
            p_intial[cone] = total_risk_per_cone[cone]*self.p_max/total_risk
        power_bound = [(0.25*self.p_max,self.p_max)]*self.n_cones
        constraints = {"type": "eq", "fun": self.power_sum_constraint}
        result = minimize(lambda power: self.cost(power, cones), p_intial, bounds=power_bound, constraints=constraints)
        self.p_optimal = result.x

        power_opti = self.p_optimal
        print(power_opti)

        self.sub.update(sample, sample_index, scene_id, power_opti)

        lidar_new = self.sub.subsamp
        count_new = self.sub.count_new

        print(count_new)
        #print(self.sub.count)

        self.filt.update(sample, sample_index, lidar_new, count_new)
        objs_scan = self.filt.object_scanned

        #print(len(objs_scan))
        #print(self.filt.count)

        return lidar_new, objs_scan

    def cones (self):
        angle_step = 360 / self.n_cones
        quadrants = [(i * angle_step, (i + 1) * angle_step) for i in range(self.n_cones)]
        return quadrants

    def get_angle_and_distance(self, cell):
        ego_pos = self.ego[self._sampleindex]
        x = cell.x
        y = cell.y
        angle = math.degrees(math.atan2((y-ego_pos[1]),(x-ego_pos[0])))
        distance = math.sqrt((y-ego_pos[1])**2 + (x-ego_pos[0])**2)
        return angle, distance
    
    def assign_cell_to_cone(self, cells):
        cones = self.cones()
        cone_cells = {i: [] for i in range(self.n_cones)}
        for cell in cells:
            angle, distance = self.get_angle_and_distance(cell)

            if distance <= self.max_range:
                for i, (start_angle, end_angle) in enumerate(cones):
                    if start_angle <= angle < end_angle:
                        cone_cells[i].append((cell, distance))
                        break

        return cone_cells


    def calc_proba(self, power,r):
        
        amp_area = np.pi*(D/2)**2
        P_s = (1/(2*np.pi*r**2))*power*erx*etx*n*amp_area
        Aov = 4*r**2*np.tan(Aovx/2)*np.tan(Aovy/2)
        P_b = (1/(2*np.pi*r**2))*phi_amb*amp_area*Aov*erx*n
        SNR = np.sqrt(Nshots*Ro**2*P_s**2)/np.sqrt(2*e*Bn*F*(Ro*(P_s+P_b)+Id)+(Bn/M**2)*(4*Kb*T/Rf + (vamp/Rf)**2))
        Prob = 0.5*math.erfc(np.sqrt(-math.log(P_false))-np.sqrt(SNR+0.5))
        return Prob

    def cost(self, power,cones):
        total_cost = 0
        for cone, cone_cells in cones.items():
            for cell,distance in cone_cells:
                prob = self.calc_proba(power[cone],distance)
                
                risk = cell.total_risk[self._sampleindex]
                total_cost+= (1-prob)*risk
        return total_cost
    def power_sum_constraint(self,power):
        return np.sum(power*T/self.n_cones) - 0.75*self.p_max*T
