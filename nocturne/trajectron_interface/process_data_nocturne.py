import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm
import dill
import os
import json
from functools import reduce
import re
import hashlib

import sys
sys.path.append('../../../Trajectron-plus-plus/trajectron')
from environment import Environment, Scene, Node, derivative_of
from utils import prediction_output_to_trajectories
from pyquaternion import Quaternion


from nocturne.trajectron_interface.trajectron_interface_utils import default_standardization, trajectory_curvature, data_columns_vehicle, data_columns_pedestrian
from nocturne.trajectron_interface.kalman_filter import NonlinearKinematicBicycle

waymo_dt = 1 / 10.
curv_0_2 = 0
curv_0_1 = 0
total = 0

def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name, non_aug_scene=scene)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        if node.type == 'PEDESTRIAN':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}

            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)
        elif node.type == 'VEHICLE':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            heading = getattr(node.data.heading, '°').copy()
            heading += alpha
            heading = (heading + np.pi) % (2.0 * np.pi) - np.pi

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

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

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep,
                        non_aug_node=node)

        scene_aug.nodes.append(node)
    return scene_aug

def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    scene_aug.map = scene.map
    return scene_aug

def extract_single_obs(env, row_obj, downsample):
    """
    returns: pd.DataFrame of all the valid data points of a specific agent
    """
    valid_indices = np.where(np.array(row_obj['valid']) == True)[0]
    ls_obs = []
    if row_obj['type'] == 'vehicle':
        agent_type = env.NodeType.VEHICLE
    elif row_obj['type'] == 'pedestrian':
        agent_type = env.NodeType.PEDESTRIAN
    else:
        # cyclist, which we dont care
        return pd.DataFrame()
    
    first_valid = valid_indices[0]
    node_id_str = str(row_obj['position'][first_valid]) + str(row_obj['goalPosition']) + str(row_obj['width']) + str(row_obj['length'])
    node_id = hashlib.sha256(node_id_str.encode('ascii')).hexdigest()[:16]
    
    total_ts = int(90 // downsample) + 1
    for i in range(total_ts):
        original_idx = i * downsample
        if original_idx in valid_indices:
            data_point = pd.DataFrame({'frame_id': i,
                                'type': agent_type,
                                'node_id': node_id,
                                'robot': False,
                                'x': row_obj['position'][original_idx]['x'],
                                'y': row_obj['position'][original_idx]['y'],
                                'z': 0.0,
                                'length': row_obj['length'],
                                'width': row_obj['width'],
                                'height': 0.0,
                                'heading': np.radians(row_obj['heading'][original_idx]),
                                'vx': row_obj['velocity'][original_idx]['x'],
                                'vy': row_obj['velocity'][original_idx]['y']}, index=[0])
            ls_obs.append(data_point)
    return reduce(lambda l, r: pd.concat([l, r]), ls_obs)

def process_scene(env, scene_path, downsample):
    scene_id = re.search(r'(tfrecord.*)(\.json)', scene_path).groups()[0]
    with open(scene_path) as f:
        tfrecord = json.load(f)
        scene_df = pd.DataFrame(tfrecord['objects'])
        ls_scene_dp_df = scene_df.apply(lambda row: extract_single_obs(env, row, downsample), axis=1).values
        data = reduce(lambda l, r: pd.concat([l, r]), ls_scene_dp_df).reset_index(drop=True)
    

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

    scene = Scene(timesteps=max_timesteps + 1, dt=waymo_dt * downsample, name=str(scene_id))

    # add map limit in Scene
    scene.xlim = (x_min, x_max)
    scene.ylim = (y_min, y_max)


    for node_id in pd.unique(data['node_id']):
        node_frequency_multiplier = 1
        node_df = data[data['node_id'] == node_id]

        if node_df['x'].shape[0] < 2:
            # print(f"skipping {node_id} reason 1 {node_df['x'].shape}")
            continue

        if not np.all(np.diff(node_df['frame_id']) == 1):
            # print('Occlusion')
            # print(f"skipping {node_id} reason 2 {np.diff(node_df['frame_id'])}")
            continue  # TODO Make better

        node_pos_values = node_df[['x', 'y']].values
        x = node_pos_values[:, 0].copy()
        y = node_pos_values[:, 1].copy()
        heading = node_df['heading'].values
        node_vel_values = node_df[['vx', 'vy']].values
        vx = node_vel_values[:, 0].copy()
        vy = node_vel_values[:, 1].copy()
        # vx = derivative_of(x, scene.dt)
        # vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        ########## DEBUG ONLY ###########
        # print(node_id)
        # print(x)
        # print(vx)
        # print(ax)
        # print(heading)
        # print(derivative_of(heading, scene.dt, radian=True))
        # print()

        if node_df.iloc[0]['type'] == env.NodeType.VEHICLE:
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
    
    return scene


def process_data(data_path, output_path, test_size, do_augment, fstart, fend, downsample):
    with open(f'{data_path}/valid_files.json') as f:
        valids = json.load(f)
        print(f'Number of valid files:{len(valids.keys())}')
    scene_fnames = [f'{data_path}/{fname}' for fname in valids.keys()][fstart:fend]
    scene_fnames.sort(key=lambda x: hashlib.sha256(x.encode('ascii')).hexdigest())

    if test_size <= 0:
        train_scenes = scene_fnames
        test_scenes = []
    else:
        train_scenes, test_scenes = train_test_split(scene_fnames, test_size=test_size)

    waymo_scene_paths = dict()
    waymo_scene_paths['train'] = train_scenes
    waymo_scene_paths['test'] = test_scenes

    for data_class in ['train', 'test']:
        env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=default_standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

        env.attention_radius = attention_radius
        env.robot_type = env.NodeType.VEHICLE
        env.scenes = []

        for scene_path in tqdm(waymo_scene_paths[data_class]):
            scene = process_scene(env, scene_path, downsample)
            if scene is not None:
                if data_class == 'train' and do_augment > 0:
                    scene.augmented = list()
                    angles = np.arange(0, 360, 60)
                    for angle in angles:
                        scene.augmented.append(augment_scene(scene, angle))
                env.scenes.append(scene)

        print(f'Processed {len(env.scenes):.2f} scenes')

        if len(env.scenes) > 0:
            data_dict_path = os.path.join(output_path, f'Waymo_Nocturne_s{fstart}_e{fend}_{data_class}_downsample{downsample}_full.pkl')
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
            print('Saved Environment!')

        global total
        global curv_0_2
        global curv_0_1
        print(f"Total Nodes: {total}")
        print(f"Curvature > 0.1 Nodes: {curv_0_1}")
        print(f"Curvature > 0.2 Nodes: {curv_0_2}")
        total = 0
        curv_0_1 = 0
        curv_0_2 = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--augment', type=int, default=1)
    parser.add_argument('--fstart', type=int, required=True)
    parser.add_argument('--fend', type=int, required=True)
    parser.add_argument('--downsample', type=int, default=1)
    args = parser.parse_args()
    process_data(args.data, args.output_path, args.test_size, args.augment, args.fstart, args.fend, args.downsample)
