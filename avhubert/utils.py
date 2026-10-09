# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import os
import torch
import random
import numpy as np
from typing import Dict, List, Optional, Tuple

def load_video(path):
    for i in range(3):
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame)
                else:
                    break
            frames = np.stack(frames)
            return frames
        except Exception:
            print(f"failed loading {path} ({i} / 3)")
            if i == 2:
                raise ValueError(f"Unable to load {path}")


class _CacheUnavailable(Exception):
    """帧缓存目录缺失或索引与 bin 不配套时抛出（内部用，调用方会回退到在线解码）。"""


class FrameCache(object):
    """预先解码好的 ROI 帧缓存（numpy 内存映射），用来省掉每样本的 mp4 解码。

    为什么要它（实测数据，单卡 3090 + 8 核配额）
    --------------------------------------------
    每个 clip 的 ``load_video`` 约 16–17 ms，拆开看是：

        open 4.1 ms | 逐帧 read 12.1 ms | cvtColor 0.6 ms | stack 0.4 ms

    而一个 4000-token 的 batch 约 80 个 clip，光解码就要 1.3 秒，
    比这一步的 GPU 前反向还贵——**训练实际是在等数据，不是等 GPU**。
    ROI 帧总量实测约 **24 GB**（2,611,599 帧 × 96×96 字节），
    预解码成一个大文件 + 内存映射后读取退化成一次 memcpy
    （页缓存命中约 0.1 ms/clip，对比在线解码约 16 ms），代价是磁盘占用。

    用法
    ----
        cache = FrameCache('/path/to/frame_cache')   # 目录含 frames.bin / index.npz
        frames = cache.get('/abs/path/to/xxx.mp4')   # 命中 -> (T,H,W) uint8；未命中 -> None

    未命中时调用方应回退到 ``load_video``，所以缓存缺条目不会让训练失败。
    缓存的键是「数据集里解析出来的视频绝对路径」，和 ``load_video`` 收到的路径一致。
    """

    def __init__(self, cache_dir):
        self.dir = cache_dir
        self._mm = None
        self._pid = None
        self._keys = None
        self._hits = 0
        self._misses = 0
        self._broken = False
        self._why = ""

    def _ensure(self):
        # DataLoader 是多进程：fork 出来的子进程要各自重新打开 memmap，
        # 所以这里按 pid 判断，而不是只在构造时打开一次。
        if self._mm is not None and self._pid == os.getpid():
            return
        if self._broken:
            # 缓存目录缺失/不配套时不要把训练搞崩：退化成"永远未命中"，
            # 由调用方回退到在线解码（只是慢一点）。只在第一次打一条警告。
            raise _CacheUnavailable(self._why)
        try:
            self._load()
        except Exception as e:                                # noqa: BLE001
            self._broken = True
            self._why = "%s（%s）" % (self.dir, e)
            print("警告：帧缓存不可用，回退到在线解码：%s" % self._why)
            raise _CacheUnavailable(self._why)

    def _load(self):
        bin_path = os.path.join(self.dir, "frames.bin")
        index_path = os.path.join(self.dir, "index.npz")
        with np.load(index_path, allow_pickle=False) as idx:
            bin_size = int(idx["bin_size"]) if "bin_size" in idx else None
            keys = idx["keys"]
            offsets = idx["offsets"]
            lengths = idx["lengths"]
            heights = idx["heights"]
            widths = idx["widths"]
            self._keys = {
                str(k): (int(o), int(l), int(h), int(w))
                for k, o, l, h, w in zip(keys, offsets, lengths, heights, widths)
            }
        # 索引与 bin 是否配套：不配套就把缓存标记为不可用，绝不带着错位的
        # offset 去读——那种情况下每个切片都能 reshape 成功，会静默读到
        # 别的 clip 的帧（见 build_frame_cache.py 里的说明）。
        actual = os.path.getsize(bin_path)
        if bin_size is not None and actual != bin_size:
            raise ValueError("frames.bin 大小 %d != 索引记录的 %d，索引与 bin 不配套"
                             % (actual, bin_size))
        self._mm = np.memmap(bin_path, dtype=np.uint8, mode="r")
        self._pid = os.getpid()

    def __len__(self):
        self._ensure()
        return len(self._keys)

    def hit_rate(self):
        tot = self._hits + self._misses
        return (float(self._hits) / tot) if tot else 0.0

    def get(self, path):
        try:
            self._ensure()
        except _CacheUnavailable:
            return None
        item = self._keys.get(path)
        if item is None:
            self._misses += 1
            return None
        off, length, h, w = item
        self._hits += 1
        return self._mm[off: off + length * h * w].reshape(length, h, w)


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        # 显式转 float32：不做的话 uint8 与 python float 相减会被提升成 **float64**，
        # 而这一串 transform 的出口在 hubert_dataset.py 里本来就要 astype(np.float32)。
        # float64 中间结果让每个 500 帧 clip 多出约 36.8 MB 的拷贝（float32 是 18.4 MB），
        # 在 8 个 DataLoader worker 里就是白白的带宽和分配器压力。
        if frames.dtype == np.uint8 or frames.dtype == np.uint16:
            frames = frames.astype(np.float32)
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)

class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class HorizontalFlip(object):
    """Flip image horizontally.
    """

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames

def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape
    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    batch_indexes, starts, ends = [], [], []
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
        vals, run_starts, run_lengths = find_runs(mask[i])
        start_indices, lengths = run_starts[vals == True], run_lengths[vals == True]
        starts.append(start_indices)
        ends.append(start_indices+lengths)
        batch_indexes.append(np.zeros([len(start_indices)])+i)
    return mask, np.concatenate(starts).astype(np.int64), np.concatenate(ends).astype(np.int64), np.concatenate(batch_indexes).astype(np.int64)

def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths
