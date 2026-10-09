# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
统计每个 clip 的视频帧数 / 音频帧数，按 rank 分片写出。

本项目的两处改动（详见 docs/代码审查与优化记录.md）：

1. ``--video-only``：纯视觉任务（``task.modalities: ["video"]``）下音频那一列
   **根本不会被读** —— hubert_dataset.py 只读 ``items[-2]``（视频帧数），
   ``load_feature`` 也只在 ``'audio' in modalities`` 时才去读 wav。
   所以该模式不再打开 48,164 个 wav，音频帧数统一写 0 当占位。
   流水线里用 ``MODALITY=video``（默认）会自动带上这个开关，
   顺带省掉整步抽音频（约 1 小时 + 3.3 GB）。
   注意：``lrs2_manifest.py:79,90`` 仍要求 ``nframes.audio`` **文件存在且行数一致**，
   所以占位文件必须照写。

2. 原实现 ``cv2.VideoCapture(...).get(CAP_PROP_FRAME_COUNT)`` **不检查能否打开**：
   打不开或截断的 mp4 会返回 0 并被原样写进 ``nframes.video``，而配置里没有
   ``min_sample_size``，0 帧样本不会被过滤，最后炸在训练时的 ``np.stack([])``
   （utils.py 里 ``Unable to load``）——报错位置离病因隔了好几个小时。
   现在：打不开、或帧数非正数的，一律计入 ``missing.list.<rank>``、并以非 0 退出，
   让流水线在第 5 步就明确报错。
"""

import cv2, math, os
from tqdm import tqdm
from scipy.io import wavfile


def count_frames(fids, audio_dir, video_dir, video_only=False):
    total_num_frames = []
    bad = []
    for fid in tqdm(fids):
        wav_fn = f"{audio_dir}/{fid}.wav"
        video_fn = f"{video_dir}/{fid}.mp4"
        cap = cv2.VideoCapture(video_fn)
        ok = cap.isOpened()
        num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
        cap.release()
        if not ok or num_frames_video <= 0:
            bad.append(fid)
            continue
        if video_only:
            num_frames_audio = 0          # 占位列，纯视觉训练不读它
        else:
            num_frames_audio = len(wavfile.read(wav_fn)[1])
        total_num_frames.append([num_frames_audio, num_frames_video])
    return total_num_frames, bad


def check(fids, audio_dir, video_dir, video_only=False):
    missing = []
    for fid in tqdm(fids):
        wav_fn = f"{audio_dir}/{fid}.wav"
        video_fn = f"{video_dir}/{fid}.mp4"
        is_file = os.path.isfile(video_fn) and (video_only or os.path.isfile(wav_fn))
        if not is_file:
            missing.append(fid)
    return missing


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='count number of frames', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, help='root dir')
    parser.add_argument('--manifest', type=str, help='a list of filenames')
    parser.add_argument('--nshard', type=int, default=1, help='number of shards')
    parser.add_argument('--rank', type=int, default=0, help='rank id')
    parser.add_argument('--video-only', action='store_true',
                        help='纯视觉模式：不读 wav，音频帧数列写 0 占位（该列不会被训练读取）')
    args = parser.parse_args()
    fids = [ln.strip() for ln in open(args.manifest).readlines()]
    print(f"{len(fids)} files")
    audio_dir, video_dir = f"{args.root}/audio", f"{args.root}/video"
    ranks = list(range(0, args.nshard))
    fids_arr = []
    num_per_shard = math.ceil(len(fids)/args.nshard)
    for rank in ranks:
        sub_fids = fids[rank*num_per_shard: (rank+1)*num_per_shard]
        if len(sub_fids) > 0:
            fids_arr.append(sub_fids)
    if args.rank >= len(fids_arr):
        open(f"{args.root}/nframes.audio.{args.rank}", 'w').write('')
        open(f"{args.root}/nframes.video.{args.rank}", 'w').write('')
    else:
        fids = fids_arr[args.rank]
        missing_fids = check(fids, audio_dir, video_dir, args.video_only)
        if len(missing_fids) > 0:
            print(f"Some audio/video files not exist, see {args.root}/missing.list.{args.rank}")
            with open(f"{args.root}/missing.list.{args.rank}", 'w') as fo:
                fo.write('\n'.join(missing_fids)+'\n')
        else:
            num_frames, bad = count_frames(fids, audio_dir, video_dir, args.video_only)
            if bad:
                print(f"{len(bad)} clips unreadable, see {args.root}/missing.list.{args.rank}")
                with open(f"{args.root}/missing.list.{args.rank}", 'w') as fo:
                    fo.write('\n'.join(bad)+'\n')
                raise SystemExit(1)
            audio_num_frames = [x[0] for x in num_frames]
            video_num_frames = [x[1] for x in num_frames]
            with open(f"{args.root}/nframes.audio.{args.rank}", 'w') as fo:
                fo.write(''.join([f"{x}\n" for x in audio_num_frames]))
            with open(f"{args.root}/nframes.video.{args.rank}", 'w') as fo:
                fo.write(''.join([f"{x}\n" for x in video_num_frames]))
