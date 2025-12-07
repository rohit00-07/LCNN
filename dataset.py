# dataset.py
import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio


def create_splits(
    data_root: str = "data",
    splits_dir: str = "splits",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Scan data_root/bona_fide and data_root/spoof, create train/val/test txt files.
    Each line in txt:   relative_path label
    Example:            bona_fide/xxx1.wav 0
    """
    os.makedirs(splits_dir, exist_ok=True)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    classes = [("bona_fide", 0), ("spoof", 1)]
    all_samples: List[Tuple[str, int]] = []

    for subdir, label in classes:
        full_dir = os.path.join(data_root, subdir)
        if not os.path.isdir(full_dir):
            raise FileNotFoundError(f"Directory not found: {full_dir}")
        for fname in os.listdir(full_dir):
            if fname.lower().endswith(".wav"):
                rel_path = os.path.join(subdir, fname)  # e.g. bona_fide/xxx.wav
                all_samples.append((rel_path, label))

    random.seed(seed)
    random.shuffle(all_samples)

    n_total = len(all_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    def _write_split(filename: str, samples: List[Tuple[str, int]]):
        with open(os.path.join(splits_dir, filename), "w") as f:
            for path, label in samples:
                f.write(f"{path} {label}\n")

    _write_split("train.txt", train_samples)
    _write_split("val.txt", val_samples)
    _write_split("test.txt", test_samples)

    print(f"Total: {n_total}, Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")


class ASVSpoofDataset(Dataset):
    """
    Reads a split file (train.txt / val.txt / test.txt),
    loads wav, resamples to 16k, computes LFCC, returns (1, F, T_frames), label.
    """
    def __init__(
        self,
        split_file: str,
        data_root: str = "data",
        sample_rate: int = 16000,
        n_lfcc: int = 60,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
    ):
        super().__init__()
        self.data_root = data_root
        self.sample_rate = sample_rate

        # Read list
        self.items: List[Tuple[str, int]] = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, label_str = line.split()
                label = int(label_str)
                self.items.append((rel_path, label))

        # torchaudio transforms
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_lfcc=n_lfcc,
            speckwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
                "center": True,
                "power": 2.0,
            },
        )

        self.resamplers = {}  # cache resamplers by original sr

    def __len__(self):
        return len(self.items)

    def _ensure_sample_rate(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        # waveform: (channels, T)
        if orig_sr == self.sample_rate:
            return waveform
        if orig_sr not in self.resamplers:
            self.resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.sample_rate,
            )
        resampler = self.resamplers[orig_sr]
        return resampler(waveform)

    def __getitem__(self, idx: int):
        rel_path, label = self.items[idx]
        full_path = os.path.join(self.data_root, rel_path)

        waveform, sr = torchaudio.load(full_path)  # (C, T)
        waveform = self._ensure_sample_rate(waveform, sr)

        # mono: average channels if needed
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # squeeze to (T) for LFCC
        wav_1d = waveform.squeeze(0)  # (T,)

        lfcc = self.lfcc(wav_1d)  # (n_lfcc, T_frames)

        # Optional: per-utterance mean-variance normalization
        mean = lfcc.mean(dim=1, keepdim=True)
        std = lfcc.std(dim=1, keepdim=True) + 1e-6
        lfcc = (lfcc - mean) / std

        # Add channel dim -> (1, F, T)
        lfcc = lfcc.unsqueeze(0)

        return lfcc, label


def asvspoof_collate_fn(batch):
    """
    Batch is a list of (feat, label), where feat: (1, F, T_i).
    We pad along time dimension to max T in the batch.
    """
    feats, labels = zip(*batch)
    batch_size = len(feats)
    _, F, _ = feats[0].shape
    max_T = max(f.shape[2] for f in feats)

    batch_feats = torch.zeros(batch_size, 1, F, max_T, dtype=feats[0].dtype)
    for i, f in enumerate(feats):
        T = f.shape[2]
        batch_feats[i, :, :, :T] = f

    batch_labels = torch.tensor(labels, dtype=torch.long)
    return batch_feats, batch_labels