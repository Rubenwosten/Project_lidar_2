
import numpy as np
import math
from scipy.optimize import minimize
#lidar_parameters:
erx = 0.9 # receiver optics effeciency
etx = 0.9 # emmitter optics effeciency
n = 0.1 #target reflectivity
D = 25*pow(10,-3) #diameter lens 25 mm
Aovx = 1/np.pi #1 graden in radialen
Aovy = 1/np.pi #1 graden in radialen
phi_amb = 37.72 #W/m^2 gekozen via tabel want test wast delta labda = 50 nm
Nshots = 1
Ro = 0.9 #We kiezen een ADP lidar met een golflengte van 903 nm
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

def calc_proba(power,r):
    
    amp_area = np.pi*(D/2)**2
    P_s = (1/(2*np.pi*r**2))*power*erx*etx*n*amp_area
    Aov = 4*r**2*np.tan(Aovx/2)*np.tan(Aovy/2)
    P_b = (1/(2*np.pi*r**2))*phi_amb*amp_area*Aov*erx*n
    SNR = np.sqrt(Nshots*Ro**2*P_s**2)/np.sqrt(2*e*Bn*F*(Ro*(P_s+P_b)+Id)+(Bn/M**2)*(4*Kb*T/Rf + (vamp/Rf)**2))
    Prob = 0.5*math.erfc(np.sqrt(-math.log(P_false))-np.sqrt(SNR+0.5))
    return Prob

pro = calc_proba(8,25)
print(pro)
e = 1080*2.997*10**8*6.626*10**-34/(900*10**-9)
print(e)
#dec lines.
'''            if self.constant_power==True:
                self.file_get()
                #print ("file complete")
                info = self.nusc.get('sample', self._sample)
                info = self.nusc.get('sample_data', info['data']['LIDAR_TOP'])
                
                sen_info = self.nusc.get('calibrated_sensor', info['calibrated_sensor_token'])
                
                info_2 = self.nusc.get('ego_pose', info['ego_pose_token'])
                
                rot_2 = np.arctan2((2*(sen_info['rotation'][0]*sen_info['rotation'][3]+sen_info['rotation'][1]*sen_info['rotation'][2])),(1-2*(sen_info['rotation'][3]**2+sen_info['rotation'][2]**2)))
                rot = np.arctan2((2*(info_2['rotation'][0]*info_2['rotation'][3]+info_2['rotation'][1]*info_2['rotation'][2])),(1-2*(info_2['rotation'][3]**2+info_2['rotation'][2]**2))) 
                
                rot_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                rot_matrix_2 = np.array([[np.cos(rot_2), -np.sin(rot_2)], [np.sin(rot_2), np.cos(rot_2)]])
                xy_lidar = np.array([sen_info['translation'][0], sen_info['translation'][1]]).reshape(-1, 1) 
                self.lidar_coor(rot_matrix, rot_matrix_2, xy_lidar )
                #print("lidar complete")
                if prnt:
                    print ("file complete")
            else:
'''

#track lines
'''             if self.constant_power == True:
                info = self.nusc.get('sample', self._sample)
                anns = info['anns']
                print(f'amount of objects within the sample = {len(anns)}')
            else:
'''