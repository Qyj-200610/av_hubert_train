#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_dataset.py -- 对预处理产物做「全量」完整性验证（不是抽样）

为什么需要：manifest 的核心契约是
    tsv 里的视频帧数(items[-2]) == landmark pkl 的条数 == ROI 视频的实际帧数
`hubert_dataset.load_audio_visual` 用 items[-2] 做样本过滤/分批，
`align_mouth.crop_patch` 用 len(landmarks) 当循环上界。
一旦这三者不一致，训练不会报错，只会静默用错长度的样本（或错的分批）。

检查项（全量 48,164 条）：
  1. landmark pkl 能否 pickle.load，且条数 == tsv 里的视频帧数
  2. ROI mp4 能否打开，且 CAP_PROP_FRAME_COUNT == tsv 里的视频帧数
  3. ROI 帧尺寸是否为 96x96
  4. 每个 split 的 tsv 行数、fid 唯一性、path 列指向的文件是否存在
  5. train/valid/test 之间是否有 fid 重叠（测试泄漏）
  6. landmark 里 None 的比例（没检到脸的帧，align_mouth 会插值填补）

用法（在实例上跑）：
    python verify_dataset.py --work-dir /hy-tmp/lrs2_data --workers 8 [--limit N]
"""
import argparse
import math
import os
import pickle
import sys
from multiprocessing import Pool


def read_lines(path):
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def check_one(args):
    """子进程：检查一条 clip 的三方一致性。"""
    fid, nf_video, video_dir, land_dir = args
    import cv2
    res = {"fid": fid, "err": [], "none_frames": 0, "n_lm": None, "n_vid": None,
           "shape": None, "all_none": False}
    # landmark
    lp = os.path.join(land_dir, fid + ".pkl")
    try:
        with open(lp, "rb") as fh:
            lm = pickle.load(fh)
    except Exception as e:
        res["err"].append("landmark 读取失败: %r" % e)
        return res
    res["n_lm"] = len(lm)
    res["none_frames"] = sum(1 for x in lm if x is None)
    # 整段都没检测到人脸：align_mouth.py 会把这种 clip 静默替换成"整帧缩放图"
    # （`if not preprocessed_landmarks: resizing` 分支），对唇读模型来说是纯噪声输入。
    # 原来的检查只统计语料级 None 帧数，看不出有多少条 clip 属于这一类，
    # 所以这里单独标出来。
    res["all_none"] = (res["n_lm"] > 0 and res["none_frames"] == res["n_lm"])
    # 视频
    vp = os.path.join(video_dir, fid + ".mp4")
    # cap 在 try 之外先声明 + finally 里释放：原实现只在"正常读完"和
    # "打不开"两条路径上 release，一旦中间抛异常（解码器出错等）句柄就泄漏；
    # 4.8 万条 × 多 worker 下有耗尽句柄的现实风险。
    cap = None
    try:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            res["err"].append("ROI 视频打不开")
            return res
        res["n_vid"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res["shape"] = (w, h)
    except Exception as e:
        res["err"].append("ROI 视频读取异常: %r" % e)
        return res
    finally:
        if cap is not None:
            cap.release()
    # 三方一致
    if nf_video is not None and res["n_lm"] != nf_video:
        res["err"].append("landmark 条数 %d != tsv 记录 %d" % (res["n_lm"], nf_video))
    if nf_video is not None and res["n_vid"] != nf_video:
        res["err"].append("ROI 帧数 %d != tsv 记录 %d" % (res["n_vid"], nf_video))
    if res["shape"] != (96, 96):
        res["err"].append("ROI 尺寸 %s != (96, 96)" % (res["shape"],))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    W = args.work_dir
    # 先检查文件**是否存在**再读：read_lines 里的 open() 没保护，缺文件会直接抛
    # FileNotFoundError，把"该跑第 5 步了"这句人话提示完全盖掉。
    for _f, _hint in (("file.list", "先跑流水线第 1 步（prepare）"),
                      ("nframes.video", "先跑流水线第 5 步（frames）")):
        _p = os.path.join(W, _f)
        if not os.path.isfile(_p):
            print("错误：找不到 %s —— %s。" % (_p, _hint))
            return 1
    fids = read_lines(os.path.join(W, "file.list"))
    # nframes.video 可能是空文件（流水线没跑到第 5 步），直接 int() 会抛
    # 一个和病因无关的异常；先给出人话提示。
    _nf_lines = read_lines(os.path.join(W, "nframes.video"))
    if not _nf_lines:
        print("错误：%s 是空文件——流水线第 5 步可能没跑完，请重跑（frames）。"
              % os.path.join(W, "nframes.video"))
        return 1
    try:
        nf_video = [int(x) for x in _nf_lines]
    except ValueError as e:
        print("错误：nframes.video 里有非整数值（%r）——文件可能被写坏，建议重跑第 5 步。" % e)
        return 1
    print("file.list %d 条, nframes.video %d 行" % (len(fids), len(nf_video)))
    if len(fids) != len(nf_video):
        print("错误：file.list 与 nframes.video 长度不一致")
        return 1
    nf_map = dict(zip(fids, nf_video))

    # ---- split / tsv 检查 ----
    print("\n=== tsv 检查 ===")
    seen = {}
    tsv_problems = []          # 结构性错误：格式坏了 / 有重复 id / 视频文件缺失 / 帧数列不符
    for split in ("train", "valid", "test", "test_mv"):
        p = os.path.join(W, "data", "%s.tsv" % split)
        if not os.path.isfile(p):
            print("  %-8s 不存在" % split)
            tsv_problems.append("%s.tsv 不存在" % split)
            continue
        with open(p, encoding="utf-8") as fh:
            root = fh.readline().rstrip("\n")
            rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
        col_ok = all(len(r) == 5 for r in rows)
        ids = [r[0] for r in rows]
        dup = len(ids) - len(set(ids))
        missing_path = 0
        bad_count = 0
        malformed = 0
        for r in rows:
            # 列数不对时直接 r[1]/r[3] 会抛 IndexError，把"tsv 格式坏了"报成
            # 一句看不懂的 traceback；这里先判定再取值。
            if len(r) != 5:
                malformed += 1
                continue
            if not os.path.isfile(r[1]):
                missing_path += 1
            try:
                if r[0] in nf_map and int(r[3]) != nf_map[r[0]]:
                    bad_count += 1
            except ValueError:
                malformed += 1
        print("  %-8s 行=%-6d root=%-3s 列数全为5=%s 重复id=%d 视频文件缺失=%d 帧数列与nframes不符=%d 格式异常=%d"
              % (split, len(rows), repr(root), col_ok, dup, missing_path, bad_count, malformed))
        # ★ 这些原来只是"打印出来"，不影响退出码 —— 于是整份 tsv 列数都不对
        #   （比如被写坏成 4 列）时脚本仍然 exit 0，调用方以为校验通过。
        #   现在汇总进 tsv_problems，最后一起决定退出码。
        if root != "/":
            tsv_problems.append("%s.tsv 首行不是 root 行（是 %r）" % (split, root))
        if not col_ok:
            tsv_problems.append("%s.tsv 有 %d 行列数不为 5" % (split, malformed))
        if dup:
            tsv_problems.append("%s.tsv 有 %d 条重复 id" % (split, dup))
        if missing_path:
            tsv_problems.append("%s.tsv 有 %d 条视频文件缺失" % (split, missing_path))
        if bad_count:
            tsv_problems.append("%s.tsv 有 %d 条帧数列与 nframes 不符" % (split, bad_count))
        seen[split] = set(ids)
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if a in seen and b in seen:
            inter = seen[a] & seen[b]
            print("  %s ∩ %s = %d %s" % (a, b, len(inter), "OK" if not inter else "★ 泄漏！"))
            if inter:
                tsv_problems.append("%s 与 %s 有 %d 条重叠（数据泄漏）" % (a, b, len(inter)))
    if "test" in seen and "test_mv" in seen:
        # 注意别用 ⊆ 之类的非 GBK 字符：Windows 控制台默认 GBK，print 会抛
        # UnicodeEncodeError（本脚本自己踩过）
        _sub = seen["test_mv"] <= seen["test"]
        print("  test_mv 是否为 test 的子集 ? %s（%d / %d）"
              % (seen["test_mv"] <= seen["test"], len(seen["test_mv"]), len(seen["test"])))

    # ---- 逐条三方一致性（全量），覆盖所有 split 出现过的 fid ----
    targets = sorted(set().union(*seen.values())) if seen else fids
    if args.limit:
        targets = targets[: args.limit]
    print("\n=== 逐条一致性检查（%d 条）===" % len(targets))
    work = [(f, nf_map.get(f), os.path.join(W, "video"), os.path.join(W, "landmark"))
            for f in targets]

    bad, total_none, shapes, all_none_fids = [], 0, {}, []
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None
    with Pool(args.workers) as pool:
        it = pool.imap_unordered(check_one, work, chunksize=64)
        if tqdm:
            it = tqdm(it, total=len(work))
        for res in it:
            total_none += res["none_frames"]
            if res.get("all_none"):
                all_none_fids.append(res["fid"])
            if res["shape"]:
                shapes[res["shape"]] = shapes.get(res["shape"], 0) + 1
            if res["err"]:
                bad.append(res)

    print("\n=== 结果 ===")
    print("  检查条数: %d" % len(targets))
    print("  有问题条数: %d" % len(bad))
    print("  尺寸分布: %s" % shapes)
    print("  无脸帧(None) 总数: %d" % total_none)
    # 整段无脸的 clip：align_mouth 把它们替换成"整帧缩放图"，对唇读是噪声输入。
    pct = (100.0 * len(all_none_fids) / len(targets)) if targets else 0.0
    print("  整段无脸的 clip: %d 条（%.2f%%）" % (len(all_none_fids), pct))
    if all_none_fids:
        print("    这类样本的 ROI 不是嘴部裁切，而是整帧缩放，建议从训练集剔除或单独评估。")
        print("    前 5 条：%s" % ", ".join(all_none_fids[:5]))
    if bad:
        print("\n  前 10 条问题样本：")
        for r in bad[:10]:
            print("    %s: %s" % (r["fid"], "; ".join(r["err"])))
    if tsv_problems:
        print("\n  tsv 结构问题：")
        for t in tsv_problems[:20]:
            print("    - %s" % t)
    if bad or tsv_problems:
        print("\n校验未通过（有问题样本 %d 条，tsv 结构问题 %d 项）。"
              % (len(bad), len(tsv_problems)))
        return 1
    print("\n所有检查通过：tsv 帧数 / landmark 条数 / ROI 实际帧数 三者一致，尺寸均为 96x96。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
