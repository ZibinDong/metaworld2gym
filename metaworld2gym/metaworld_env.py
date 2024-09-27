from typing import List, Optional

import gym
import numpy as np
from gym import spaces

import metaworld

from .point_cloud_generator import PointCloudGenerator

TASK_BOUDNS = {
    "default": [-0.5, -1.5, -0.795, 1, -0.4, 100],
}
TASK_LIST = [
    "basketball",
    "bin-picking",
    "button-press",
    "button-press-topdown",
    "button-press-topdown-wall",
    "button-press-wall",
    "coffee-button",
    "coffee-pull",
    "coffee-push",
    "dial-turn",
    "disassemble",
    "door-lock",
    "door-open",
    "door-unlock",
    "drawer-close",
    "drawer-open",
    "faucet-close",
    "faucet-open",
    "hammer",
    "handle-press",
    "handle-press-side",
    "handle-pull",
    "handle-pull-side",
    "shelf-place",
    "soccer",
    "stick-push",
    "sweep",
    "sweep-into",
    "window-close",
    "window-open",
]


def point_cloud_sampling(
    point_cloud: np.ndarray,
    num_points: int,
    method: str = "fps",
):
    """
    support different point cloud sampling methods
    point_cloud: (N, 6), xyz+rgb or (N, 3), xyz
    """
    if num_points == "all":  # use all points
        return point_cloud

    if point_cloud.shape[0] <= num_points:
        point_cloud_dim = point_cloud.shape[-1]
        point_cloud = np.concatenate(
            [
                point_cloud,
                np.zeros((num_points - point_cloud.shape[0], point_cloud_dim)),
            ],
            axis=0,
        )
        return point_cloud

    if method == "uniform":
        # uniform sampling
        sampled_indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[sampled_indices]
    elif method == "fps":
        N = point_cloud.shape[0]
        centroids = np.zeros((num_points, point_cloud.shape[-1]), dtype=np.float32)
        idx = np.zeros(num_points, dtype=np.int64)

        idx[0] = np.random.randint(0, N)
        centroids[0] = point_cloud[idx[0]]

        min_distances = np.full(N, np.inf, dtype=np.float32)
        dist = np.sum((point_cloud[:, :3] - centroids[0, :3]) ** 2, axis=-1)
        min_distances = np.minimum(min_distances, dist)

        for m in range(1, num_points):
            idx[m] = np.argmax(min_distances)
            centroids[m] = point_cloud[idx[m]]
            dist = np.sum((point_cloud[:, :3] - centroids[m, :3]) ** 2, axis=-1)
            min_distances = np.minimum(min_distances, dist)

        point_cloud = centroids

    else:
        raise NotImplementedError(f"point cloud sampling method {method} not implemented")

    return point_cloud


class MetaWorldEnv(gym.Env):
    """Simple MetaWorld interface adapted from DP3 https://github.com/YanjieZe/3D-Diffusion-Policy.

    Args:
        task_name (str):
            Task name from MetaWorld task list. Use `env.TASK_LIST` to get the list of available tasks.

        observation_meta (List[str]):
            List of observation space to include in the environment. Defaults to ("image", "depth", "agent_pos", "point_cloud", "full_state")

        render_device (str):
            Device to render on. Defaults to "cuda:0".

        use_point_crop (bool):
            Whether to use point cloud cropping. Defaults to True.

        num_points (int):
            Number of points to sample from the point cloud. Defaults to 1024.

        image_size (int):
            Size of the image to render. Defaults to 224.
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self,
        task_name: str,
        observation_meta: List[str] = (
            "image",
            "depth",
            "agent_pos",
            "point_cloud",
            "full_state",
        ),
        render_device: str = "cuda:0",
        use_point_crop: bool = True,
        num_points: int = 1024,
        image_size: int = 224,
    ):
        super(MetaWorldEnv, self).__init__()
        self.observation_meta = observation_meta
        if "-v2" not in task_name:
            task_name = task_name + "-v2-goal-observable"

        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        self.env._freeze_rand_vec = False

        self.env.sim.model.cam_pos[2] = [0.6, 0.295, 0.8]
        self.env.sim.model.vis.map.znear = 0.1
        self.env.sim.model.vis.map.zfar = 1.5

        dict_observation_space = dict()
        self.image_size = None
        self.device_id = None
        self.pc_generator = None
        self.use_point_crop = None
        self.num_points = None
        assert len(observation_meta) > 0, "Observation space should not be empty."

        if "image" in observation_meta:
            self.image_size = image_size
            self.device_id = int(render_device.split(":")[-1])
            dict_observation_space["image"] = spaces.Box(
                low=0,
                high=255,
                shape=(3, image_size, image_size),
                dtype=np.uint8,
            )
        if "depth" in observation_meta:
            self.image_size = image_size
            self.device_id = int(render_device.split(":")[-1])
            self.pc_generator = PointCloudGenerator(sim=self.env.sim, cam_names=["corner2"], img_size=image_size)
            dict_observation_space["depth"] = spaces.Box(
                low=0,
                high=np.inf,
                shape=(image_size, image_size),
                dtype=np.float32,
            )
        if "agent_pos" in observation_meta:
            dict_observation_space["agent_pos"] = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        if "point_cloud" in observation_meta:
            self.use_point_crop = use_point_crop
            self.num_points = num_points
            self.pc_generator = PointCloudGenerator(sim=self.env.sim, cam_names=["corner2"], img_size=image_size)
            dict_observation_space["point_cloud"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(num_points, 3), dtype=np.float32
            )
            x_rad = np.deg2rad(61.4)
            y_rad = np.deg2rad(-7)
            self.pc_transform = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(x_rad), np.sin(x_rad)],
                    [0, -np.sin(x_rad), np.cos(x_rad)],
                ]
            ) @ np.array(
                [
                    [np.cos(y_rad), 0, np.sin(y_rad)],
                    [0, 1, 0],
                    [-np.sin(y_rad), 0, np.cos(y_rad)],
                ]
            )
            self.pc_scale = np.array([1, 1, 1])
            self.pc_offset = np.array([0, 0, 0])
            if task_name in TASK_BOUDNS:
                x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS[task_name]
            else:
                x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS["default"]
            self.min_bound = [x_min, y_min, z_min]
            self.max_bound = [x_max, y_max, z_max]
        if "full_state" in observation_meta:
            dict_observation_space["full_state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32)

        self.episode_length = self._max_episode_steps = 200
        self.action_space = self.env.action_space

    @property
    def TASK_LIST(self):
        return TASK_LIST

    def get_robot_state(self):
        eef_pos = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env._get_site_pos("rightEndEffector"),
            self.env._get_site_pos("leftEndEffector"),
        )
        return np.concatenate([eef_pos, finger_right, finger_left])

    def get_rgb(self):
        # cam names: ('topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV')
        img = self.env.sim.render(
            width=self.image_size,
            height=self.image_size,
            camera_name="corner2",
            device_id=self.device_id,
        )
        return img

    def get_point_cloud(self, use_rgb=True):
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(device_id=self.device_id)

        if not use_rgb:
            point_cloud = point_cloud[..., :3]

        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale

        if self.pc_offset is not None:
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset

        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]

        point_cloud = point_cloud_sampling(point_cloud, self.num_points, "fps")

        depth = depth[::-1]

        return point_cloud, depth

    def get_visual_obs(self):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()

        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            "image": obs_pixels,
            "depth": depth,
            "agent_pos": robot_state,
            "point_cloud": point_cloud,
        }
        return obs_dict

    def get_observation(self, raw_state: Optional[np.ndarray] = None):
        obs_dict = {}
        if "image" in self.observation_meta:
            obs_pixels = self.get_rgb()
            if obs_pixels.shape[0] != 3:  # make channel first
                obs_pixels = obs_pixels.transpose(2, 0, 1)
            obs_dict["image"] = obs_pixels
        if "depth" in self.observation_meta or "point_cloud" in self.observation_meta:
            point_cloud, depth = self.get_point_cloud()
            if "depth" in self.observation_meta:
                obs_dict["depth"] = depth
            if "point_cloud" in self.observation_meta:
                obs_dict["point_cloud"] = point_cloud
        if "agent_pos" in self.observation_meta:
            obs_dict["agent_pos"] = self.get_robot_state()
        if "full_state" in self.observation_meta:
            obs_dict["full_state"] = raw_state
        return obs_dict

    def step(self, action: np.array):
        raw_state, reward, done, env_info = self.env.step(action)
        self.cur_step += 1
        obs_dict = self.get_observation(raw_state)
        done = done or self.cur_step >= self.episode_length
        return obs_dict, reward, done, env_info

    def reset(self):
        self.env.reset()
        self.env.reset_model()
        raw_state = self.env.reset()
        self.cur_step = 0
        obs_dict = self.get_observation(raw_state)
        return obs_dict

    def render(self, mode="rgb_array"):
        img = self.get_rgb()
        return img

    def close(self):
        return self.env.close()


def make(
    env_name: str,
    observation_meta=(
        "image",
        "depth",
        "agent_pos",
        "point_cloud",
        "full_state",
    ),
    render_device: str = "cuda:0",
    use_point_crop: bool = True,
    num_points: int = 1024,
    image_size: int = 224,
):
    return MetaWorldEnv(env_name, observation_meta, render_device, use_point_crop, num_points, image_size)
