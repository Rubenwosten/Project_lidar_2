import numpy as np
from nuscenes import NuScenes



# this class designates a severity function according to identified objects and their orientation
class severity:

    def orientation_assign(angle):
        if 315<= angle or angle < 45 :
            return "front"
        elif 45 <= angle < 135: 
            return "side"
        elif 135 <= angle < 225:
            return "rear"
        elif 225 <= angle <315:
            return "side"


    def factor(traffic_participant, participant_facing, participant_position, ego_facing, self_x, self_y, detected): 

        #vehicle.car [0.9687457208122074, 0.0, 0.0, -0.24805589774894815] [587.422, 1654.32, 1.126] [0.5780193421828153, -0.002390749587151425, 0.012332644429184535, -0.8159263632585605] 600 1600
              

        traffic_participant_f = {       #the dictionary which defines severity factor according to traffic participant
        "human.pedestrian.adult": {
            "score": 1,       #the factor based on category
            "orientation": 0            #wether the orientation of the participant is relevant or not
        },
        "human.pedestrian.child": {
            "score": 1,
            "orientation": 0
        },
        "human.pedestrian.constructi": {
            "score": 1,
            "orientation": 0
        },
        "human.pedestrian.personal_m": {
            "score": 1,
            "orientation": 0
        },
        "human.pedestrian.police_off": {
            "score": 1,
            "orientation": 0
        },
        "movable_object.barrier": {
            "score": 0.369369369,
            "orientation": 0
        },
        "movable_object.debris": {
            "score": 0.36036036,
            "orientation": 0
        },
        "movable_object.pushable_pullable": {
            "score": 0.36036036,
            "orientation": 0
        },
        "movable_object.trafficcone": {
            "score": 0.36036036,
            "orientation": 0
        },
        "static_object.bicycle_rack": {
            "score": 0.522522523,
            "orientation": 0
        },
        "vehicle.bicycle": {
            "score": 1,
            "orientation": 1
        },
        "vehicle.bus.bendy": {
            "score": 1.0,
            "orientation": 1
        },
        "vehicle.bus.rigid": {
            "score": 1.0,
            "orientation": 1
        },
        "vehicle.car": {
            "score": 0.810811,
            "orientation": 1
        },
        "vehicle.construction": {
            "score": 1.0,
            "orientation": 1
        },
        "vehicle.motorcycle": {
            "score": 0.765766,
            "orientation": 1
        },
        "vehicle.trailer": {
            "score": 1.318,
            "orientation": 1
        },
        "vehicle.truck": {
            "score": 1.0,
            "orientation": 1
        }
        }

        orientation_f={             #the dictionary defining the orientation factor vs ego vehicle
            "front": 0.8,
            "side": 1,
            "rear": 0.4
        }

        ego_orientation_f={         # the orientation factor of the ego vegicle
            "front": 0.8,
            "side": 1,
            "rear": 0.4
        }
    

        #evaluating the positions of both vehicles
        participant_x, participant_y, z = participant_position
        v_e_p= np.array([participant_x - self_x, participant_y - self_y])   #generating a vector from ego to participant
        uv_e_p= v_e_p / np.linalg.norm (v_e_p)                              #converting to a unit vector

        #evaluating the orientations of both vehicles
        ego_facing_a = np.arctan2((2*(ego_facing[0]*ego_facing[3]+ego_facing[1]* ego_facing[2])),(1-2*(ego_facing[3]**2+ego_facing[2]**2)))
        ego_facing_v = np.array(np.cos(ego_facing_a), np.sin(ego_facing_a))
        ego_angle = np.arccos(np.dot(ego_facing_v, uv_e_p)[0])
        ego_orientation= severity.orientation_assign(ego_angle)

        if detected:
            participant_facing = (0,0,0,0)

        participant_facing_a = np.arctan2((2*(participant_facing[0]*participant_facing[3]+participant_facing[1]* participant_facing[2])),(1-2*(participant_facing[3]**2+participant_facing[2]**2)))
        participant_facing_v = np.array(np.cos(participant_facing_a), np.sin(participant_facing_a))
        participant_angle = np.arccos(np.dot(participant_facing_v, -uv_e_p)[0])
        orientation = severity.orientation_assign(participant_angle)
            
        
        
        



        #extracting all values according to the traffic participant
        participant_score= traffic_participant_f[traffic_participant]["score"]
        io=traffic_participant_f[traffic_participant]["orientation"]
        o_factor= orientation_f.get(orientation, 1)
        e_o_factor= ego_orientation_f.get(ego_orientation, 1)

        
        
        #severity calculation function
        if io == 1 and detected:
            sev = participant_score * o_factor * e_o_factor
            #print(sev)
        else:
            sev = participant_score * e_o_factor
            #print(sev)
        return sev
