from pathlib import Path

import numba
import numpy as np
import torch
import torch.utils.data
import zarr


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray, sequence_length: int, pad_before: int = 0, pad_after: int = 0, debug: bool = True
) -> np.ndarray:
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0  # episode start index
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]  # episode end index
        episode_length = end_idx - start_idx  # episode length

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


class SequenceSampler:
    def __init__(
        self,
        root_dir: str,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        zero_padding: bool = False,
    ):
        super().__init__()
        assert sequence_length >= 1

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        self.root = zarr.open(root_dir / "metaworld.zarr", mode="r")

        if keys is None:
            keys = list(self.root["data"].keys())

        episode_ends = self.root["meta"]["episode_ends"][:]

        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.task_ends = [task_info["end_idx"] for task_info in self.root.meta.attrs["task_info"]]
        self.task_ids = [task_info["task_id"] for task_info in self.root.meta.attrs["task_info"]]
        self.task_t5_emb = self.root.meta["lang_t5_emb"][:]
        self.task_t5_mask = self.root.meta["lang_t5_mask"][:]

        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.zero_padding = zero_padding

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.root["data"][key]
            sample = input_arr[buffer_start_idx:buffer_end_idx]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(shape=(self.sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
                if not self.zero_padding:
                    if sample_start_idx > 0:
                        data[:sample_start_idx] = sample[0]
                    if sample_end_idx < self.sequence_length:
                        data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        task_idx = np.searchsorted(self.task_ends, buffer_end_idx, side="right")
        result["task_id"] = self.task_ids[task_idx]
        result["task_t5_emb"] = self.task_t5_emb[task_idx]
        result["task_t5_mask"] = self.task_t5_mask[task_idx]
        return result


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        zero_padding: bool = False,
    ):
        super().__init__()
        self.sampler = SequenceSampler(
            root_dir=root_dir,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=keys,
            zero_padding=zero_padding,
        )

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        return self.sampler.sample_sequence(idx)


def get_dataset(
    root_dir: str,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
    keys=None,
    zero_padding: bool = False,
):
    return SequenceDataset(
        root_dir=root_dir,
        sequence_length=sequence_length,
        pad_before=pad_before,
        pad_after=pad_after,
        keys=keys,
        zero_padding=zero_padding,
    )
