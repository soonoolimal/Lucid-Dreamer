import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = pathlib.Path(__file__).parents[2]


@dataclass
class HDF5:
    observations: np.ndarray   # (N, C=3, H, W) uint8
    actions: np.ndarray        # (N,)/(N, act_dim) if discrete/continuous
    rewards: np.ndarray        # (N,)
    returns_to_go: np.ndarray  # (N,)
    done_idxs: np.ndarray      # (E,) cumulative episode-end indices
    timesteps: np.ndarray      # (N,) in-episode step index
    labels: np.ndarray         # (N,) dynamics label (= dy_type)
    num_episodes: int
    max_ep_len: int            # actual max episode length (from data, not HDF5 timeout attr)
    act_dim: int               # 1/(1, cont_dim) if discrete/mixed
    n_act: int                 # actions.max()+1/act_dim if discrete/continuous
    is_discrete: bool


def make_hdf5_paths(
    scn: str,
    ds_type: str,                            # 'dreamer' | 'random'
    seed: int,
    n_dynamics: int,
    timestamps: Optional[List[str]] = None,  # one per dy_type; auto-detect if None
) -> List[pathlib.Path]:
    """Reconstruct HDF5 paths from per-dy_type timestamps.

    Path pattern:
        data/{scn}/{ds_type}/{timestamp}_dy{dy_type}_s{seed}.hdf5

    If timestamps is None, picks the latest file per dy_type automatically.
    """
    if ds_type not in ('dreamer', 'random'):
        raise ValueError(f"ds_type must be 'dreamer' or 'random', got {ds_type!r}")
    data_dir = ROOT / 'data' / scn / ds_type
    paths = []
    for dy_type in range(n_dynamics):
        if timestamps is not None:
            paths.append(data_dir / f'{timestamps[dy_type]}_dy{dy_type}_s{seed}.hdf5')
        else:
            candidates = sorted(data_dir.glob(f'*_dy{dy_type}_s{seed}.hdf5'))
            if not candidates:
                raise FileNotFoundError(
                    f'No HDF5 found for dy_type={dy_type}, seed={seed} in {data_dir}'
                )
            paths.append(candidates[-1])

    for dy_type, p in enumerate(paths):
        print(f'[HDF5] dy{dy_type}: {p}')

    return paths


def load_hdf5_datasets(
    paths: List[pathlib.Path],
    gamma: float,
    train_ratio: float,
    valid_ratio: float,
) -> Tuple[List[HDF5], List[HDF5], List[HDF5]]:
    """Load each HDF5 file and split into train/valid/test per dy_type."""
    datasets = []
    for path in paths:
        label = int(pathlib.Path(path).stem.split('_dy')[-1].split('_s')[0])
        datasets.append(_load_single(str(path), label=label, gamma=gamma))

    splits = [_split(ds, train_ratio, valid_ratio) for ds in datasets]
    train = [s[0] for s in splits]
    valid = [s[1] for s in splits]
    test = [s[2] for s in splits]

    return train, valid, test


def _load_single(path: str, label: int, gamma: float) -> HDF5:
    with h5py.File(path, 'r') as hf:
        observations = hf['observations'][:]  # (N, H, W, 3)
        actions = hf['actions'][:]            # (N, 1) or (N, 1+cont_dim)
        rewards = hf['rewards'][:]            # (N,)
        timeouts = hf['timeouts'][:]          # (N,)
        num_episodes = int(hf.attrs['num_episodes'])
        act_dim = int(hf.attrs['act_dim'])
        num_transitions = int(hf.attrs['num_transitions'])

    if rewards.shape[0] != num_transitions:
        raise ValueError(
            f'{path}: transition count mismatch '
            f'(rewards={rewards.shape[0]}, attr={num_transitions})'
        )

    labels = np.full(rewards.shape[0], label, dtype=np.int64)

    done_idxs = (np.nonzero(timeouts > 0.5)[0] + 1).astype(np.int64)
    if len(done_idxs) != num_episodes:
        raise ValueError(
            f'{path}: episode count mismatch '
            f'(found {len(done_idxs)} in timeouts, attr={num_episodes})'
        )

    observations = np.transpose(observations, (0, 3, 1, 2))

    if actions.ndim == 2 and actions.shape[1] == 1:
        actions = np.squeeze(actions, axis=1).astype(np.int64)
    is_discrete = actions.ndim == 1

    n_act = int(actions.max()) + 1 if is_discrete else act_dim

    rewards = rewards.astype(np.float32)
    returns_to_go = _compute_rtg(rewards, done_idxs, gamma)

    ep_starts = np.concatenate([[0], done_idxs[:-1]])
    ep_lens = done_idxs - ep_starts
    max_ep_len = int(ep_lens.max())
    timesteps = np.concatenate([np.arange(l, dtype=np.int64) for l in ep_lens])

    return HDF5(
        observations=observations,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go.astype(np.float32),
        done_idxs=done_idxs,
        timesteps=timesteps,
        labels=labels,
        num_episodes=num_episodes,
        max_ep_len=max_ep_len,
        act_dim=act_dim,
        n_act=n_act,
        is_discrete=bool(is_discrete),
    )


def _split(
    ds: HDF5,
    train_ratio: float,
    valid_ratio: float,
) -> Tuple[HDF5, HDF5, HDF5]:
    n = ds.num_episodes
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = n - n_train - n_valid

    if n_train * n_valid * n_test == 0:
        raise ValueError(
            f'Not enough episodes for train/valid/test split: {n} '
            f'(n_train={n_train}, n_valid={n_valid}, n_test={n_test})'
        )

    def ep_slice(ep_start: int, ep_end: int) -> HDF5:
        ts_start = int(ds.done_idxs[ep_start - 1]) if ep_start > 0 else 0
        ts_end = int(ds.done_idxs[ep_end - 1])
        done_idxs = ds.done_idxs[ep_start:ep_end] - ts_start
        return HDF5(
            observations=ds.observations[ts_start:ts_end],
            actions=ds.actions[ts_start:ts_end],
            rewards=ds.rewards[ts_start:ts_end],
            returns_to_go=ds.returns_to_go[ts_start:ts_end],
            done_idxs=done_idxs,
            timesteps=ds.timesteps[ts_start:ts_end],
            labels=ds.labels[ts_start:ts_end],
            num_episodes=ep_end - ep_start,
            max_ep_len=ds.max_ep_len,
            act_dim=ds.act_dim,
            n_act=ds.n_act,
            is_discrete=ds.is_discrete,
        )

    return (
        ep_slice(0, n_train),
        ep_slice(n_train, n_train + n_valid),
        ep_slice(n_train + n_valid, n),
    )


def _compute_rtg(rewards: np.ndarray, done_idxs: np.ndarray, gamma: float) -> np.ndarray:
    rtg = np.zeros_like(rewards, dtype=np.float32)
    start = 0
    for end in done_idxs:
        r = rewards[start:end]
        if np.isclose(gamma, 1.0, rtol=1e-9, atol=1e-9):
            rtg[start:end] = np.cumsum(r[::-1])[::-1]
        else:
            out = np.empty_like(r)
            running = 0.0
            for i in range(len(r) - 1, -1, -1):
                running = r[i] + gamma * running
                out[i] = running
            rtg[start:end] = out
        start = end
    return rtg


class OfflineDataset(Dataset):
    def __init__(self, ds: HDF5, seq_len: int):
        if ds.num_episodes == 0:
            raise ValueError('Dataset has no episodes')

        ep_starts = np.concatenate([[0], ds.done_idxs[:-1]])
        ep_lens = ds.done_idxs - ep_starts

        valid = ep_lens >= seq_len
        n_dropped = int((~valid).sum())
        if n_dropped > 0:
            print(f'[OfflineDataset] dropping {n_dropped}/{len(valid)} episode(s) shorter than seq_len={seq_len}')
        ep_starts = ep_starts[valid]
        ep_lens = ep_lens[valid]
        if len(ep_lens) == 0:
            raise ValueError(f'No episodes with length >= seq_len={seq_len}')

        self.observations = ds.observations
        self.actions = ds.actions
        self.rewards = ds.rewards
        self.returns_to_go = ds.returns_to_go
        self.timesteps = ds.timesteps
        self.labels = ds.labels
        self.seq_len = seq_len
        self.act_dtype = torch.long if ds.is_discrete else torch.float32

        # stride=1 guarantees every window is exactly seq_len long without padding
        # so padding mask is always full
        stride = 1
        self.mask = torch.ones(seq_len, dtype=torch.long)
        windows_per_ep = (ep_lens - seq_len) // stride + 1
        self._cum_windows = np.concatenate([[0], np.cumsum(windows_per_ep)])
        self._ep_starts = ep_starts

    def __len__(self) -> int:
        return int(self._cum_windows[-1])

    def __getitem__(self, idx: int) -> dict:
        ep_idx = int(np.searchsorted(self._cum_windows, idx, side='right')) - 1
        window_in_ep = idx - self._cum_windows[ep_idx]
        ts_start = self._ep_starts[ep_idx] + window_in_ep
        sl = slice(ts_start, ts_start + self.seq_len)

        return dict(
            observations=torch.from_numpy(self.observations[sl]).float().div(255.0),
            actions=torch.from_numpy(self.actions[sl]).to(self.act_dtype),
            rewards=torch.from_numpy(self.rewards[sl]).float().unsqueeze(-1),
            returns_to_go=torch.from_numpy(self.returns_to_go[sl]).float().unsqueeze(-1),
            timesteps=torch.from_numpy(self.timesteps[sl]).long(),
            mask=self.mask,
            labels=torch.from_numpy(self.labels[sl]).long(),
        )


def make_dataloader(
    ds_list: List[HDF5],
    seq_len: int,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        ConcatDataset([OfflineDataset(ds, seq_len) for ds in ds_list]),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
