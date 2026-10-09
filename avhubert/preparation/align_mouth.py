# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

## Based on: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/crop_mouth_from_video.py

""" Crop Mouth ROIs from videos for lipreading

性能优化（本项目加的，实测数据见 docs/训练加速与阻塞修复记录.md）
--------------------------------------------------------------------
官方实现在每个 clip 上做了两件很贵的事，实测单 clip 约 1.2 秒：

1. **重采样用 skimage.transform.warp**（对每一帧做相似变换到 256x256）。
   实测 **535 ms/clip**，是整个唇部 ROI 阶段的主要开销。
   优化：**保留 skimage 只做"估计"**（`estimate_transform('similarity')`，很便宜，
   而且保证几何与官方逐个像素一致），把"重采样"换成 `cv2.warpAffine`（121 ms/clip，
   **4.4x**）。
   注意方向：`cv2.warpAffine` 默认就把 M 当作 src→dst 的正向映射（内部自己求逆），
   所以直接传 `tform.params[:2, :]`，**不要**再 `invertAffineTransform` —— 双重求逆
   会让画面整体错位（第一版就踩了这个坑，ROI 平均差 80 个灰度级）。
   保真度实测：平均差 **0.50** 个灰度级、最大 2–3、**没有一个像素差超过 2**。

2. **逐帧写 PNG 再让 ffmpeg concat 回读**（350 ms/clip，全量会产生 240 万个临时 PNG）。
   优化：把裁剪好的帧直接通过管道喂给 ffmpeg rawvideo（244 ms/clip）。
   PNG 是无损的，所以编码器收到的像素**完全一样**，产物等价。

两个优化都可以用命令行关掉、退回官方行为以便逐位对齐：
    --warp-backend skimage --write-backend png
"""

import os,pickle,shutil,tempfile
import math
import cv2
import glob
import subprocess
import argparse
import numpy as np
from collections import deque
import cv2
from skimage import transform as tf
from tqdm import tqdm

# -- Landmark interpolation:
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

# -- 重采样 / 写视频后端（由命令行设置，见文件头说明）
WARP_BACKEND = "cv2"
WRITE_BACKEND = "pipe"

# -- Face Transformation
def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    if WARP_BACKEND == 'cv2':
        # 直接用 skimage 估计出来的 src->dst 矩阵；cv2 默认按正向映射处理，无需再求逆
        M = tform.params[:2, :].astype(np.float32)
        warped = cv2.warpAffine(img, M, std_size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        return warped, tform
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # warp
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform

def apply_transform(transform, img, std_size):
    if WARP_BACKEND == 'cv2':
        M = transform.params[:2, :].astype(np.float32)
        return cv2.warpAffine(img, M, std_size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255  # note output from warp is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped

def get_frame_count(filename):
    cap = cv2.VideoCapture(filename)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def read_video(filename):
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):                                                 
        ret, frame = cap.read() # BGR
        if ret:                      
            yield frame                                                    
        else:                                                              
            break                                                         
    cap.release()

# -- Crop
def cut_patch(img, landmarks, height, width, threshold=5):

    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:                                                
        center_y = height                                                    
    if center_y - height < 0 - threshold:                                    
        raise Exception('too much bias in height')                           
    if center_x - width < 0:                                                 
        center_x = width                                                     
    if center_x - width < 0 - threshold:                                     
        raise Exception('too much bias in width')                            
                                                                             
    if center_y + height > img.shape[0]:                                     
        center_y = img.shape[0] - height                                     
    if center_y + height > img.shape[0] + threshold:                         
        raise Exception('too much bias in height')                           
    if center_x + width > img.shape[1]:                                      
        center_x = img.shape[1] - width                                      
    if center_x + width > img.shape[1] + threshold:                          
        raise Exception('too much bias in width')                            
                                                                             
    cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img

def write_video_ffmpeg(rois, target_path, ffmpeg):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    fps = 25
    rois = np.asarray(rois)
    if os.path.isfile(target_path):
        os.remove(target_path)

    try:
        return _write_video_ffmpeg_inner(rois, target_path, ffmpeg, fps)
    except Exception:
        # 关键：失败时不能留下半个文件。ffmpeg 不会自己删掉写坏的输出，
        # 而主循环用"目标已存在就跳过"来做续跑（align_mouth.py 的 resume 逻辑），
        # 于是这个残缺文件会被当成"已完成"永久保留，后续 count_frames 只看
        # os.path.isfile 也照样通过，帧数就成了脏数据。
        if os.path.isfile(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass
        raise


def _write_video_ffmpeg_inner(rois, target_path, ffmpeg, fps):
    if WRITE_BACKEND == 'pipe':
        # 直接把帧以 rawvideo 喂给 ffmpeg：省掉"逐帧写 PNG + concat 回读"这一整轮
        # 临时文件 I/O（全量会产生约 240 万个临时 PNG）。PNG 无损，编码器收到的
        # 像素与原来完全一致，所以产物等价。
        h, w = rois.shape[1], rois.shape[2]
        # 必须带上官方那套编码参数（尤其 -q:v 1）：少了它虽然"编码器收到的像素一样"，
        # 但编码器用的是 CRF 20 而不是 qscale=1，产物会整体差一档。
        # 实测（与官方 png 路径逐帧对比）：少 -q:v 1 时平均差 2.1 灰度级、34% 像素差>2；
        # 补上之后两条路径的产物才对齐。
        cmd = [ffmpeg, "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h),
               "-r", str(fps), "-i", "-", "-q:v", "1", "-crf", "20", "-y", target_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for roi in rois:
                if roi.ndim == 2:                      # 灰度帧也兼容一下
                    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                proc.stdin.write(np.ascontiguousarray(roi, dtype=np.uint8).tobytes())
        finally:
            proc.stdin.close()
            rc = proc.wait()
        # 原实现完全不检查 ffmpeg 的返回码，失败时只会"悄悄没有产物"；
        # 这里改成显式报错（上层会记录并按失败处理），避免静默丢样本。
        if rc != 0 or not os.path.isfile(target_path):
            raise RuntimeError("ffmpeg 写视频失败(pipe, rc=%s): %s" % (rc, target_path))
        return

    # ---- 官方原实现：逐帧 PNG + concat demuxer ----
    decimals = 10
    tmp_dir = tempfile.mkdtemp()
    for i_roi, roi in enumerate(rois):
        cv2.imwrite(os.path.join(tmp_dir, str(i_roi).zfill(decimals)+'.png'), roi)
    list_fn = os.path.join(tmp_dir, "list")
    with open(list_fn, 'w') as fo:
        fo.write("file " + "'" + tmp_dir+'/%0'+str(decimals)+'d.png' + "'\n")
    ## ffmpeg
    cmd = [ffmpeg, "-f", "concat", "-safe", "0", "-i", list_fn, "-q:v", "1", "-r", str(fps), '-y', '-crf', '20', target_path]
    pipe = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    # rm tmp dir
    shutil.rmtree(tmp_dir)
    if pipe.returncode != 0 or not os.path.isfile(target_path):
        raise RuntimeError("ffmpeg 写视频失败(png, rc=%s): %s" % (pipe.returncode, target_path))
    return

def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading Pre-processing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video-direc', default=None, help='raw video directory')
    parser.add_argument('--landmark-direc', default=None, help='landmark directory')
    parser.add_argument('--filename-path', help='list of detected video and its subject ID')
    parser.add_argument('--save-direc', default=None, help='the directory of saving mouth ROIs')
    # -- mean face utils
    parser.add_argument('--mean-face', type=str, help='reference mean face (download from: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/20words_mean_face.npy)')
    # -- mouthROIs utils
    parser.add_argument('--crop-width', default=96, type=int, help='the width of mouth ROIs')
    parser.add_argument('--crop-height', default=96, type=int, help='the height of mouth ROIs')
    parser.add_argument('--start-idx', default=48, type=int, help='the start of landmark index')
    parser.add_argument('--stop-idx', default=68, type=int, help='the end of landmark index')
    parser.add_argument('--window-margin', default=12, type=int, help='window margin for smoothed_landmarks')
    parser.add_argument('--ffmpeg', type=str, help='ffmpeg path')
    parser.add_argument('--rank', type=int, help='rank id')
    parser.add_argument('--nshard', type=int, help='number of shards')
    # 性能相关（默认用经过实测校验的快路径；--warp-backend skimage --write-backend png
    # 可完全退回官方行为，用于逐位对齐）
    parser.add_argument('--warp-backend', default='cv2', choices=['cv2', 'skimage'],
                        help='图像重采样后端：cv2=快 4.4x（平均差 0.5 灰度级），skimage=官方原实现')
    parser.add_argument('--write-backend', default='pipe', choices=['pipe', 'png'],
                        help='写视频后端：pipe=直接喂 ffmpeg（快），png=官方逐帧 PNG+concat')

    args = parser.parse_args()
    return args


def crop_patch(video_pathname, landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, window_margin, start_idx, stop_idx, crop_height, crop_width):

    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """

    frame_idx = 0
    num_frames = get_frame_count(video_pathname)
    frame_gen = read_video(video_pathname)
    margin = min(num_frames, window_margin)
    # trans 只在 len(q_frame)==margin 的分支里被赋值。若本片的 landmark 条数
    # 少于 margin（例如 OpenCV 报的帧数与 skvideo 实际解出的帧数不一致、
    # 或 CAP_PROP_FRAME_COUNT 为 0），下面 flush 尾巴的分支会用到未赋值的 trans
    # 而抛 UnboundLocalError（官方原实现就有这个隐患）。
    # 这里显式初始化并给出明确失败（返回 None，由调用方断言报错），
    # 而不是让它以"变量未定义"的形式炸掉。
    trans = None
    while True:
        try:
            frame = frame_gen.__next__() ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img( smoothed_landmarks[stablePntsIDs, :],
                                           mean_face_landmarks[stablePntsIDs, :],
                                           cur_frame,
                                           STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append( cut_patch( trans_frame,
                                        trans_landmarks[start_idx:stop_idx],
                                        crop_height//2,
                                        crop_width//2,))
        if frame_idx == len(landmarks)-1:
            if trans is None:
                # 帧数不足 margin，从未估计出相似变换：无法对齐裁切，明确失败
                return None
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform( trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append( cut_patch( trans_frame,
                                            trans_landmarks[start_idx:stop_idx],
                                            crop_height//2,
                                            crop_width//2,))
            return np.array(sequence)
        frame_idx += 1
    return None


def landmarks_interpolate(landmarks):
    
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


if __name__ == '__main__':
    args = load_args()

    # 性能后端（默认 cv2 + pipe = 快路径；要逐位对齐官方就传 skimage + png）
    WARP_BACKEND = args.warp_backend
    WRITE_BACKEND = args.write_backend

    # -- mean face utils
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(args.mean_face)
    stablePntsIDs = [33, 36, 39, 42, 45]

    lines = open(args.filename_path).readlines()
    fids = [ln.strip() for ln in lines]
    num_per_shard = math.ceil(len(fids)/args.nshard)
    start_id, end_id = num_per_shard*args.rank, num_per_shard*(args.rank+1)
    fids = fids[start_id: end_id]

    # 单个 clip 出错不要立刻炸掉整段（本阶段全量约 2 小时）：先记录下来跑完其余的，
    # 最后以非 0 退出码结束，让流水线的 run_ranks_parallel 能发现并报错。
    failed = []
    for filename_idx, filename in enumerate(tqdm(fids)):

        video_pathname = os.path.join(args.video_direc, filename+'.mp4')

        landmarks_pathname = os.path.join(args.landmark_direc, filename+'.pkl')
        dst_pathname = os.path.join(args.save_direc, filename+'.mp4')

        assert os.path.isfile(video_pathname), "File does not exist. Path input: {}".format(video_pathname)
        assert os.path.isfile(landmarks_pathname), "File does not exist. Path input: {}".format(landmarks_pathname)

        if os.path.exists(dst_pathname):
            continue

        try:
            landmarks = pickle.load(open(landmarks_pathname, 'rb'))

            # -- pre-process landmarks: interpolate frames not being detected.
            preprocessed_landmarks = landmarks_interpolate(landmarks)

            if not preprocessed_landmarks:
                print(f"resizing {filename}")
                frame_gen = read_video(video_pathname)
                frames = [cv2.resize(x, (args.crop_width, args.crop_height)) for x in frame_gen]
                write_video_ffmpeg(frames, dst_pathname, args.ffmpeg)
                continue

            # -- crop
            sequence = crop_patch(video_pathname, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, window_margin=args.window_margin, start_idx=args.start_idx, stop_idx=args.stop_idx, crop_height=args.crop_height, crop_width=args.crop_width)
            assert sequence is not None, "cannot crop from {}.".format(filename)

            # -- save
            os.makedirs(os.path.dirname(dst_pathname), exist_ok=True)
            write_video_ffmpeg(sequence, dst_pathname, args.ffmpeg)
        except Exception as e:  # noqa: BLE001 - 记录后继续，最后统一报错
            failed.append((filename, repr(e)))
            if len(failed) <= 10:
                print("rank %d 处理失败: %s -> %s" % (args.rank, filename, e))

    print('Done. rank %d: 成功 %d, 失败 %d' % (args.rank, len(fids) - len(failed), len(failed)))
    if failed:
        print("以下样本失败（最多列 10 条）：")
        for fn, err in failed[:10]:
            print("  %s: %s" % (fn, err))
        raise SystemExit(1)
