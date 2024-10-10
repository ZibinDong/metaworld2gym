# Gym wrapper and automatic expert demonstration collection for Meta-World

This repository contains a gym wrapper and an automatic data collection script for Meta-World and is modified from [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy/tree/master/third_party/Metaworld). Compared to DP3-version, this repository makes the following changes:
- Re-write the gym environment and the automatic data collection script to make them more user-friendly.
- Add a funcition to get `torch.utils.data.Dataset` from the collected zarr data for convenient loading.
- Add T5 language encoder to support language-conditioned policy learning.
- Take it as a standalone Meta-World helper from DP3.

## 🤩 Installation
```bash
>>> git clone https://github.com/ZibinDong/metaworld2gym.git
>>> cd metaworld2gym
>>> pip install -e .
```
(Optional) The environment requires `pytorch3d` to do fast fps. For those who do not want to install the whole `pytorch3d`, you can install a simplified version cleaned in [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy/tree/master/third_party/Metaworld), which has only the necessary functions for fast fps. 

```bash
>>> cd pytorch3d_simplified
>>> pip install -e .
```

## 🤩 Usage

### Environment:
Create a Meta-World environment with the following code:

```python
import metaworld2gym

env = metaworld2gym.make("door-open")
obs = env.reset()
act = env.action_space.sample()
next_obs, reward, done, info = env.step(act)

for k, v in obs.items():
    print(k, v.shape)

# output:
# image (3, 224, 224)
# depth (224, 224)
# point_cloud (1024, 6)
# agent_pos (9,)
# full_state (39,)
```

You can specify the observation space by passing the `observation_meta` argument:

```python
env = metaworld2gym.make("door-open", observation_meta=["image", "depth"])
obs = env.reset()

for k, v in obs.items():
    print(k, v.shape)

# output:
# image (3, 224, 224)
# depth (224, 224)
```

You can also specify the image size and the number of points in the point cloud by passing the `image_size` and `num_points` arguments:

```python
env = metaworld2gym.make("door-open", 
    image_size=128, num_points=512, 
    observation_meta=["image", "depth", "point_cloud"])
obs = env.reset()

for k, v in obs.items():
    print(k, v.shape)

# output:
# image (3, 128, 128)
# depth (128, 128)
# point_cloud (512, 6)
```

Use `env.TASK_LIST` to get the list of available tasks.

### Data Collection:

Collect expert demonstrations with the following code:

```python
metaworld2gym.collect_dataset(
    num_episodes=100,
    root_dir="data",
    render_device="cuda:0",
    num_point_clouds=1024,
    image_size=224,
    chunk_size=10,
    T5_model="google-t5/t5-base"  # or the local path to the T5 model
)
```
This will collect 100 episodes of expert demonstrations for each of the 30 tasks in Meta-World and save the data in the `data` directory. The data will be saved in the [zarr](https://zarr.readthedocs.io/en/stable/) format. `render_device` specifies the device to use for rendering. `num_point_clouds` and `image_size` specify the number of points in the point cloud and the size of the image, respectively. `chunk_size` specifies the number of episodes to save in each chunk.

>>> **NOTE:** `T5_model` is designed to be optional. However, this feature is not implemented yet.

After collecting the data, you can load it with the following code:

```python
dataset = metaworld2gym.get_dataset(
    root_dir="data",
    sequence_length=4,
    pad_before=0,
    pad_after=0,
)
```

Here `dataset` is an instance of `torch.utils.data.Dataset`. You can use it with `torch.utils.data.DataLoader` to create a data loader for training. The batch data is a dictionary with `action`, all `observation_meta` keys and the values are tensors of shape `(batch_size, sequence_length, ...)`, and some task information like `task_id`,  `task_t5_emb` and `task_t5_mask`. All T5 embeddings are of shape `(batch_size, 32, 768)`, you can use `task_t5_mask` to mask out the padding tokens. The `sequence_length`, `pad_before`, and `pad_after` arguments specify the sequence length and the number of frames to pad before and after each sequence.


## 🤩 References
- [Meta-World](https://meta-world.github.io/)
- [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy/tree/master/third_party/Metaworld)

## 🤩 Issues

If you have any questions or issues, please open an issue on this repository. Or you can contact me via email: `zibindong@outlook.com`.
