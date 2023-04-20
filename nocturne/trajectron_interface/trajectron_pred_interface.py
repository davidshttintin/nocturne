import numpy as np
import pandas as pd

import sys
sys.path.append('../../Trajectron-plus-plus/trajectron')
from environment import Environment, Scene, Node, derivative_of, DoubleHeaderNumpyArray
from utils import prediction_output_to_trajectories
from pyquaternion import Quaternion


from .trajectron_interface_utils import default_standardization, trajectory_curvature, data_columns_vehicle, data_columns_pedestrian, load_model
from .kalman_filter import NonlinearKinematicBicycle

class TrajectronBaseInterface:
    def __init__(self, dt):
        self.env = self.make_env()
        self.dt = dt

    def make_env(self):
        env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=default_standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

        env.attention_radius = attention_radius
        env.robot_type = env.NodeType.VEHICLE
        env.scenes = []
        return env

class TrajectronInterface(TrajectronBaseInterface):
    def __init__(self, dt, model_dir='../../data/models/models_04_Apr_2022_11_54_58_int_ee', cp=12):
        super(TrajectronInterface, self).__init__(dt)
        self.scene = None # currently we have one interface per scene
        self.model_dir = model_dir # loaded after building the first scene
        self.cp = cp
        self.curvature = {
            'curv_0_2': 0,
            'curv_0_1': 0,
            'total': 0
        }

    def convert_one_obs(self, ts, agent_id, obs_nocturne: np.ndarray):
        """
        Ingest one nocturne observation (one agent at one timestep)
        and convert to pd.Series
        args: obs []
        """
        data_point = pd.Series({'frame_id': ts,
                                'type': self.env.NodeType.VEHICLE,
                                'node_id': str(agent_id),
                                'robot': False,
                                'x': obs_nocturne[0],
                                'y': obs_nocturne[1],
                                'z': 0.0,
                                'length': obs_nocturne[4],
                                'width': obs_nocturne[5],
                                'height': 0.0,
                                'heading': obs_nocturne[2]})
        
        return data_point

    def build_scene_offline(self, batched_obs_nocturne):
        """
        args: batched_obs_nocturne: List[Tuple(ts, agent_id, obs_nocturne)]
        """
        scene_id = len(self.env.scenes) # unique id strictly incrementing from 0

        data = pd.DataFrame(columns=['frame_id',
                                    'type',
                                    'node_id',
                                    'robot',
                                    'x', 'y', 'z',
                                    'length',
                                    'width',
                                    'height',
                                    'heading'])

        for ts, agent_id, obs_nocturne in batched_obs_nocturne:
            data = data.append(self.convert_one_obs(ts, agent_id, obs_nocturne), ignore_index=True)

        if len(data.index) == 0:
            return None

        data.sort_values('frame_id', inplace=True)
        max_timesteps = data['frame_id'].max()

        x_min = np.round(data['x'].min() - 50)
        x_max = np.round(data['x'].max() + 50)
        y_min = np.round(data['y'].min() - 50)
        y_max = np.round(data['y'].max() + 50)

        data['x'] = data['x'] - x_min
        data['y'] = data['y'] - y_min

        scene = Scene(timesteps=max_timesteps + 1, dt=dt, name=str(scene_id))

        # add map limit in Scene
        scene.xlim = (x_min, x_max)
        scene.ylim = (y_min, y_max)


        for node_id in pd.unique(data['node_id']):
            node_frequency_multiplier = 1
            node_df = data[data['node_id'] == node_id]

            if node_df['x'].shape[0] < 2:
                continue

            if not np.all(np.diff(node_df['frame_id']) == 1):
                # print('Occlusion')
                continue  # TODO Make better

            node_values = node_df[['x', 'y']].values
            x = node_values[:, 0]
            y = node_values[:, 1]
            heading = node_df['heading'].values
            if node_df.iloc[0]['type'] == self.env.NodeType.VEHICLE and not node_id == 'ego':
                # Kalman filter Agent
                vx = derivative_of(x, scene.dt)
                vy = derivative_of(y, scene.dt)
                velocity = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1)

                filter_veh = NonlinearKinematicBicycle(dt=scene.dt, sMeasurement=1.0)
                P_matrix = None
                for i in range(len(x)):
                    if i == 0:  # initalize KF
                        # initial P_matrix
                        P_matrix = np.identity(4)
                    elif i < len(x):
                        # assign new est values
                        x[i] = x_vec_est_new[0][0]
                        y[i] = x_vec_est_new[1][0]
                        heading[i] = x_vec_est_new[2][0]
                        velocity[i] = x_vec_est_new[3][0]

                    if i < len(x) - 1:  # no action on last data
                        # filtering
                        x_vec_est = np.array([[x[i]],
                                            [y[i]],
                                            [heading[i]],
                                            [velocity[i]]])
                        z_new = np.array([[x[i + 1]],
                                        [y[i + 1]],
                                        [heading[i + 1]],
                                        [velocity[i + 1]]])
                        x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
                            x_vec_est=x_vec_est,
                            u_vec=np.array([[0.], [0.]]),
                            P_matrix=P_matrix,
                            z_new=z_new
                        )
                        P_matrix = P_matrix_new

                curvature, pl, _ = trajectory_curvature(np.stack((x, y), axis=-1))
                if pl < 1.0:  # vehicle is "not" moving
                    x = x[0].repeat(max_timesteps + 1)
                    y = y[0].repeat(max_timesteps + 1)
                    heading = heading[0].repeat(max_timesteps + 1)
                self.curvature['total'] += 1
                if pl > 1.0:
                    if curvature > .2:
                        self.curvature['curv_0_2'] += 1
                        node_frequency_multiplier = 3*int(np.floor(self.curvature['total']/self.curvature['curv_0_2']))
                    elif curvature > .1:
                        self.curvature['curv_0_1'] += 1
                        node_frequency_multiplier = 3*int(np.floor(self.curvature['total']/self.curvature['curv_0_1']))

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            if node_df.iloc[0]['type'] == self.env.NodeType.VEHICLE:
                v = np.stack((vx, vy), axis=-1)
                v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
                heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
                heading_x = heading_v[:, 0]
                heading_y = heading_v[:, 1]

                data_dict = {('position', 'x'): x,
                            ('position', 'y'): y,
                            ('velocity', 'x'): vx,
                            ('velocity', 'y'): vy,
                            ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                            ('acceleration', 'x'): ax,
                            ('acceleration', 'y'): ay,
                            ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                            ('heading', 'x'): heading_x,
                            ('heading', 'y'): heading_y,
                            ('heading', '°'): heading,
                            ('heading', 'd°'): derivative_of(heading, scene.dt, radian=True)}
                node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
            else:
                data_dict = {('position', 'x'): x,
                            ('position', 'y'): y,
                            ('velocity', 'x'): vx,
                            ('velocity', 'y'): vy,
                            ('acceleration', 'x'): ax,
                            ('acceleration', 'y'): ay}
                node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

            node = Node(node_type=node_df.iloc[0]['type'], node_id=node_id, data=node_data, frequency_multiplier=node_frequency_multiplier)
            node.first_timestep = node_df['frame_id'].iloc[0]
            if node_df.iloc[0]['robot'] == True:
                node.is_robot = True
                scene.robot = node

            scene.nodes.append(node)
        
        
        self.env.scenes.append(scene)
        self.scene = scene
        return scene
    
    def load_model(self):
        print('loading model frozen checkpoint')
        self.model, self.hyperparam = load_model(self.model_dir, self.env, ts=self.cp)

    def make_pred(
        self, 
        timesteps: np.ndarray,  
        num_samples=1,
        min_future_timesteps=0, 
        min_history_timesteps=1):
        return self.model.predict(self.scene, timesteps, self.hyperparam['prediction_horizon'], num_samples, min_future_timesteps, min_history_timesteps)
    
    def get_features(
        self, 
        timesteps: np.ndarray,  
        num_samples=1,
        min_future_timesteps=0, 
        min_history_timesteps=1):
        return self.model.get_features(self.scene, timesteps, self.hyperparam['prediction_horizon'], num_samples, min_future_timesteps, min_history_timesteps)
    
    def prediction_out_to_trajectory(self, pred_out_dict):
        return prediction_output_to_trajectories(
            pred_out_dict, 
            self.scene.dt, 
            max_h=self.hyperparam['maximum_history_length'], 
            ph=self.hyperparam['prediction_horizon']
        )

class VehicleRollingData:
    def __init__(self, first_ts, dt):
        self.dt = dt
        self.reset(first_ts)
    
    def reset(self, new_first_ts):
        self.x = np.nan
        self.y = np.nan
        self.vx = np.nan
        self.vy = np.nan
        self.ax = np.nan
        self.ay = np.nan
        self.theta = np.nan
        self.dtheta = np.nan
        self.last_valid_beginning = new_first_ts
        self.last_ts = new_first_ts
    
    def update_data(self, ts, newx, newy, newtheta):
        if ts == self.last_valid_beginning:
            pass
        elif ts == self.last_ts + 1:
            newvx = (newx - self.x) / self.dt
            newvy = (newy - self.y) / self.dt
            newdtheta = (newtheta - self.theta) / self.dt
            if ts >= self.last_valid_beginning + 2:
                self.ax = (newvx - self.vx) / self.dt
                self.ay = (newvy - self.vy) / self.dt
            self.vx = newvx
            self.vy = newvy
            self.dtheta = newdtheta
        else:
            # skipping frames
            print(f'skipping frames at {ts}, last ts{self.last_ts}, valid begin{self.last_valid_beginning}')
            self.reset(ts)
        self.x = newx
        self.y = newy
        self.theta = newtheta
        self.last_ts = ts
    
    def has_valid_action(self, ts):
        return self.last_ts == ts and self.last_ts >= self.last_valid_beginning + 2
    
    def latest_entry(self):
        return np.array([[
            self.x,
            self.y,
            self.vx,
            self.vy,
            self.ax,
            self.ay,
            self.theta,
            self.dtheta
        ]])


class TrajectronEnvOnlineInterface(TrajectronBaseInterface):
    def __init__(self, dt):
        super(TrajectronEnvOnlineInterface, self).__init__(dt)
        self.agent_nodes = {}
        self.agent_data = {}
        self.agent_occurences = {}
    
    def injest_one_obs(self, ts, agent_id, obs_nocturne: np.ndarray):
        """
        Ingest one nocturne observation (one agent at one timestep)
        and save the data
        args: obs []
        """
        psuedo_data = DoubleHeaderNumpyArray(
            np.array([]),
            [('position', 'x'),
            ('position', 'y'),
            ('velocity', 'x'),
            ('velocity', 'y'),
            ('acceleration', 'x'),
            ('acceleration', 'y'),
            ('heading', '°'),
            ('heading', 'd°')])
        if agent_id not in self.agent_nodes:
            self.agent_occurences[agent_id] = 1
            self.agent_nodes[agent_id] = Node(
                node_type=self.env.NodeType.VEHICLE,
                node_id=str(agent_id),
                data=psuedo_data,
                length=obs_nocturne[4],
                width=obs_nocturne[5],
                first_timestep=ts
                )
            
            self.agent_data[agent_id] = VehicleRollingData(first_ts=ts, dt=self.dt)
        elif ts > self.agent_data[agent_id].last_ts + 1:
            self.agent_occurences[agent_id] += 1
            self.agent_nodes[agent_id] = Node(
                node_type=self.env.NodeType.VEHICLE,
                node_id=f'{agent_id}_{self.agent_occurences[agent_id]}',
                data=psuedo_data,
                length=obs_nocturne[4],
                width=obs_nocturne[5],
                first_timestep=ts
                )
            
            self.agent_data[agent_id] = VehicleRollingData(first_ts=ts, dt=self.dt)
        self.agent_data[agent_id].update_data(ts, obs_nocturne[0], obs_nocturne[1], obs_nocturne[2])
    
    def get_latest_input_dict(self, ts):
        input_dict = {}
        for agent_id, agent_node in self.agent_nodes.items():
            rolling_data = self.agent_data[agent_id]
            if rolling_data.has_valid_action(ts):
            #if rolling_data.last_ts == ts:
                input_dict[agent_node] = rolling_data.latest_entry()
        
        return input_dict
