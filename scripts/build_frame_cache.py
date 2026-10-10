#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
build_frame_cache.py -- 把 ROI 视频预解码成「一个大文件 + 索引」，供训练时内存映射读取

为什么值得做（实测，单卡 3090 + 8 核配额）
------------------------------------------
训练时每个 clip 都要现解 mp4，``avhubert/utils.py:load_video`` 实测约 16–17 ms：

    open 4.1 ms | 逐帧 read 12.1 ms | cvtColor 0.6 ms | stack 0.4 ms

而一个 4000-token 的 batch 约 80 个 clip → 仅解码就 ~1.3 秒，比 GPU 前反向还贵。
**训练是在等数据，不是等 GPU。** 但 ROI 帧总量很小：
实测总帧数 2,611,599 × 96×96 字节 ≈ **24 GB**（本机 100 GB 的 /hy-tmp 放得下），
预解码后读取退化成一次 memcpy（页缓存命中约 0.1 ms/clip，对比在线解码约 16 ms）。

设计要点
--------
1. 逐位复现训练路径：直接调用 ``avhubert.utils.load_video``（同一个灰度转换、
   同一套 cv2 解码），所以缓存帧与在线解码**逐位一致**，不是"近似"。
2. 键（key）与数据集里解析出的视频绝对路径**完全一致**：
   数据集是 ``os.path.join(tsv 首行 root, tsv 里的视频列)``（见
   ``hubert_dataset.py:load_audio_visual`` 与 ``load_video``）。本脚本优先读
   ``<work>/data/*.tsv`` 来生成键，保证一致；没有 tsv 时退回按
   ``<work>/video/<fid>.mp4`` 构造（两种情况下 root 都是 "/"）。
3. 多进程 + 流式写入：边算边追加，内存不随数据量增长。
4. 未命中的样本在训练时会自动回退到在线解码，所以缓存不全会降速但不会出错。

用法
----
    python scripts/build_frame_cache.py --work-dir /hy-tmp/lrs2_data \\
        --out /hy-tmp/lrs2_data/frame_cache --workers 8 --verify 20

产出：
    <out>/frames.bin   所有 clip 的帧首尾相接（uint8）
    <out>/index.npz    keys / offsets / lengths / heights / widths
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np


def _load_one(args):
    """子进程入口：解一个视频，返回 (key, frames)。

    注意必须吞掉异常：``load_video`` 解不出来时会 **raise**（utils.py 里重试 3 次后
    抛 ValueError），如果让它冒出去，pool.imap_unordered 会把整个构建中断——
    48,164 条里坏 1 条就前功尽弃，而下面的 failed 统计也永远走不到。
    返回 None 由主进程记入失败清单即可。
    """
    key, path = args
    from avhubert.utils import load_video
    try:
        return key, load_video(path)
    except Exception as e:                                   # noqa: BLE001
        return key, None


def collect_keys(work_dir):
    """按数据集的方式生成 (key, 视频路径) 列表。"""
    data_dir = os.path.join(work_dir, "data")
    pairs, seen = [], set()
    tsvs = [os.path.join(data_dir, s + ".tsv") for s in ("train", "valid", "test")]
    tsvs = [p for p in tsvs if os.path.isfile(p)]
    if tsvs:
        # 与 load_audio_visual 完全一致：首行是 root，视频列是 items[1]
        for tsv in tsvs:
            with open(tsv) as f:
                root = f.readline().strip()
                for line in f:
                    items = line.rstrip("\n").split("\t")
                    if len(items) < 3:
                        continue
                    key = os.path.join(root, items[1])
                    if key not in seen:
                        seen.add(key)
                        pairs.append((key, key))
        print("从 %d 个 tsv 收集到 %d 个视频路径（root 取自 tsv 首行）" % (len(tsvs), len(pairs)))
    else:
        with open(os.path.join(work_dir, "file.list")) as f:
            fids = [ln.strip() for ln in f if ln.strip()]
        for fid in fids:
            path = os.path.join(work_dir, "video", fid + ".mp4")
            if path not in seen:
                seen.add(path)
                pairs.append((path, path))
        print("data/*.tsv 不存在，改从 file.list 构造 %d 个视频路径" % len(pairs))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True, help="预处理工作目录（含 video/ 与 file.list）")
    ap.add_argument("--out", required=True, help="输出缓存目录")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", type=int, default=0,
                    help="完成后随机抽 N 条，验证缓存与在线解码逐位一致")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（调试用）")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    pairs = collect_keys(args.work_dir)
    missing = [k for k, p in pairs if not os.path.isfile(p)]
    if missing:
        print("警告：%d 个视频文件不存在（前 3 个：%s）" % (len(missing), missing[:3]))
        pairs = [(k, p) for k, p in pairs if os.path.isfile(p)]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print("没有可处理的视频，退出")
        return 1

    os.makedirs(args.out, exist_ok=True)
    bin_path = os.path.join(args.out, "frames.bin")
    index_path = os.path.join(args.out, "index.npz")

    keys, offsets, lengths, heights, widths = [], [], [], [], []
    offset = 0
    t0 = time.time()
    failed = []

    # 两个文件必须一起换新：offset 是按 imap_unordered 的**完成顺序**分配的，
    # 所以每次构建的 key->offset 映射都不一样。如果只删 frames.bin 而留下旧的
    # index.npz，中途被杀就会得到"旧索引 + 半截且顺序不同的新 bin"，
    # 而 FrameCache 只会按 offset 切片——**每个 offset 都是 h*w 的整数倍，
    # 于是总能 reshape 成功，静默读到别的 clip 的帧**。
    #
    # 做法：**先写到 .new 临时名**，全部成功后再依次 os.replace 发布。
    # 这样"构建失败"不会破坏上一份可用缓存（原实现是开跑前先 rm 掉旧的两个文件，
    # 于是任何一次失败——包括磁盘满、被 kill——都会连累已经建好的 24 GB 缓存）。
    # 发布顺序是"先 bin 后 index"：万一在这两步之间被杀，读者会看到
    # "新 bin + 旧 index"，而 FrameCache 会用索引里的 bin_size 校验并拒绝这一对，
    # 不会静默读到错帧。
    tmp_bin_path = bin_path + ".new"
    tmp_index_path = index_path + ".partial"
    for p in (tmp_bin_path, tmp_index_path):
        if os.path.exists(p):
            os.remove(p)

    try:
        from tqdm import tqdm
        prog = tqdm(total=len(pairs), desc="解码")
    except Exception:
        prog = None

    # 解码 + 写临时 bin。整段包在 try/except 里：磁盘满（OSError）、Ctrl-C、
    # 或任何未预料的异常都会带着 ~24 GB 的半截 frames.bin.new 退出，把磁盘占住，
    # 下一次重试前都放不下——原来只有下面三个显式 return 1 的分支会清理。
    try:
        with open(tmp_bin_path, "wb") as fo, Pool(args.workers) as pool:
            for key, frames in pool.imap_unordered(_load_one, pairs, chunksize=8):
                if frames is None or frames.ndim != 3:
                    failed.append(key)
                    continue
                t, h, w = frames.shape
                fo.write(frames.tobytes())
                keys.append(key)
                offsets.append(offset)
                lengths.append(t)
                heights.append(h)
                widths.append(w)
                offset += t * h * w
                if prog is not None:
                    prog.update(1)
    except BaseException:
        if prog is not None:
            prog.close()
        for _p in (tmp_bin_path, tmp_index_path):
            if os.path.exists(_p):
                try:
                    os.remove(_p)
                except OSError:
                    pass
        print("构建中断，已清理临时文件（旧的缓存保持不变）：%s" % tmp_bin_path)
        raise
    if prog is not None:
        prog.close()

    dt = time.time() - t0
    if failed:
        print("警告：%d 条解码失败（前 3 个：%s）" % (len(failed), failed[:3]))
    if not keys:
        print("没有任何样本成功，退出（旧的缓存保持不变）")
        if os.path.exists(tmp_bin_path):
            os.remove(tmp_bin_path)
        return 1

    heights, widths = set(heights), set(widths)
    if len(heights) != 1 or len(widths) != 1:
        # 原来是 assert：python -O 下会被整个去掉，这里改成显式报错
        print("错误：帧尺寸不一致：%s / %s（旧的缓存保持不变）" % (heights, widths))
        if os.path.exists(tmp_bin_path):
            os.remove(tmp_bin_path)
        return 1
    if os.path.getsize(tmp_bin_path) != offset:
        print("错误：bin 写入字节数不符（%d != %d），放弃发布"
              % (os.path.getsize(tmp_bin_path), offset))
        os.remove(tmp_bin_path)
        return 1

    # ---- 原子发布索引 ----
    # 坑（实测踩过，而且正是它让"原子发布"这版修复失效）：
    #   np.savez(<**字符串**文件名>, ...) 在文件名不以 .npz 结尾时会**自动追加 .npz**。
    #   原来写的是 index_path + ".tmp" = "index.npz.tmp"，于是 numpy 实际写出的是
    #   "index.npz.tmp.npz"，紧跟的 os.replace("index.npz.tmp", "index.npz") 必然
    #   FileNotFoundError → 整个缓存构建中止 → 训练静默回退到在线解码（真的发生过：
    #   见 results/04_原始产物/训练与评测_全流程日志.log 里的
    #   "警告：帧缓存构建失败，改为在线解码继续训练"）。
    #   修法：传**文件对象**给 np.savez —— 这时 numpy 不会动文件名。
    #   （tmp_index_path 已在上面统一定义为 index_path + ".partial"）
    with open(tmp_index_path, "wb") as fh:
        np.savez(
            fh,
            keys=np.array(keys),
            offsets=np.array(offsets, dtype=np.int64),
            lengths=np.array(lengths, dtype=np.int64),
            heights=np.full(len(keys), heights.pop(), dtype=np.int64),
            widths=np.full(len(keys), widths.pop(), dtype=np.int64),
            # 把 bin 的实际字节数写进索引：读取端据此校验"索引与 bin 是否配套"，
            # 避免用到不完整/不匹配的一对文件
            bin_size=np.int64(offset),
            n_frames=np.int64(sum(lengths)),
        )
    os.replace(tmp_bin_path, bin_path)       # 原子发布 bin
    os.replace(tmp_index_path, index_path)   # 再原子发布索引（读者用 bin_size 校验配对）

    if not os.path.isfile(index_path):
        print("错误：索引发布失败，%s 不存在" % index_path)
        return 1

    total_frames = int(sum(lengths))
    print("完成：%d 条 clip / %d 帧 / %.1f MB / %.1f 秒（%.1f clips/s）"
          % (len(keys), total_frames, offset / 1e6, dt, len(keys) / dt))
    print("  %s" % bin_path)
    print("  %s" % index_path)

    if args.verify:
        from avhubert.utils import load_video, FrameCache
        cache = FrameCache(args.out)
        rng = np.random.RandomState(0)
        idxs = rng.choice(len(keys), size=min(args.verify, len(keys)), replace=False)
        bad = 0
        for i in idxs:
            key = keys[i]
            cached = cache.get(key)
            live = load_video(key)
            if cached is None or cached.shape != live.shape or not np.array_equal(cached, live):
                bad += 1
        print("逐位一致性校验：抽 %d 条，不一致 %d 条 %s"
              % (len(idxs), bad, "OK" if bad == 0 else "FAIL"))
        return 0 if bad == 0 else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
