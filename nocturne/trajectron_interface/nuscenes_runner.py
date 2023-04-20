import sys
sys.path.append('../../Trajectron-plus-plus/trajectron')
from environment import Environment, Scene, Node
from cfgs.config import ERR_VAL
import numpy as np
import dill
from nocturne import ObjectType, CollisionType, Vector2D

STATES = {"PEDESTRIAN": {"position": ["x", "y"], "velocity": ["x", "y"], "acceleration": ["x", "y"]}, "VEHICLE": {"position": ["x", "y"], "velocity": ["x", "y"], "acceleration": ["x", "y"], "heading": ["\u00b0", "d\u00b0"]}}

class NuScenesRunner:
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            env = dill.load(f, encoding='latin1')
        self.env : Environment = env
    
    def load_one_scenario(self, scenario_fname: int):
        """
        scenario_fname: index of the scene
        """
        self.sim = NuScenesSimulation(self.env.scenes[scenario_fname])
        self.scenario = self.sim.scenario

class NuScenesSimulation:
    def __init__(self, scene):
        self._clock = 0
        self.scenario = NuScenesScenario(scene)
    
    def step(self, dt):
        assert np.isclose(dt, 0.5), f'unsupported nuscene step dt={dt}'
        self._clock += 5
        self.scenario.set_clock(self._clock)

class NuScenesScenario:
    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self._clock = 0
        self.sdc_obj = NuScenesObject(self.scene.robot)
        self.other_objs = [
            NuScenesObject(n) for n in self.scene.nodes if (n is not self.scene.robot) and (n.type == 'VEHICLE')
        ]
        self.sdc_obj.set_target_position(
            self.expert_position(self.sdc_obj, 185)
        )
    
    def set_clock(self, ts):
        self._clock = ts
    
    def waymo_to_nuscenes_time(self, ts):
        assert ts % 5 == 0, f'unsupported nuscene step ts={ts}'
        assert ts // 5 <= 37, f'sdc does not exist for ts={ts}'
        return ts // 5
    
    def _query_expert_state(self, obj, ts):
        idx = self.waymo_to_nuscenes_time(ts)
        input_dict = self.scene.get_clipped_input_dict(idx, STATES)
        if obj.node not in input_dict:
            return np.array([ERR_VAL, ERR_VAL, ERR_VAL, ERR_VAL])
        nuscenes_states = input_dict[obj.node][0]
        speed = np.sqrt(nuscenes_states[2] ** 2 + nuscenes_states[3] ** 2)
        nocturne_states = np.array([
            nuscenes_states[0],
            nuscenes_states[1],
            nuscenes_states[6],
            speed,
            obj.length,
            obj.width
        ])
        return nocturne_states
    
    def expert_position(self, obj, ts):
        nocturne_states = self._query_expert_state(obj, ts)
        return Vector2D.from_numpy(nocturne_states[:2])
    
    def expert_speed(self, obj, ts):
        return self._query_expert_state(obj, ts)[3]
    
    def expert_heading(self, obj, ts):
        return self._query_expert_state(obj, ts)[2]
    
    def obj_gt_state(self, obj):
        return self._query_expert_state(obj, self._clock)
    
    def sdc(self):
        return self.sdc_obj
    
    def objects(self):
        return self.other_objs + [self.sdc_obj]

class NuScenesObject:
    def __init__(self, node: Node) -> None:
        self.node = node
        self.expert_control = True
        self.collided = False
        self.collision_type = CollisionType.UNCOLLIDED
        self.type = ObjectType.VEHICLE
        self.id = node.id
        self.length = 4 # hardcoded becuase Trajectron++ pkl generation code forgot to include size
        self.width = 1.7 
        self.target_position = None
    
    def set_target_position(self, target_pos_vec):
        self.target_position = target_pos_vec