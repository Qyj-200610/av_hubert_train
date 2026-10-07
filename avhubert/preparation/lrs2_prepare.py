#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lrs2_prepare.py -- AV-HuBERT LRS2(main) 数据准备

官方仓库只提供了 LRS3 / VoxCeleb2 的预处理脚本（avhubert/preparation/lrs3_prepare.py）。
本脚本把该流程适配到 LRS2 的 **main** 数据集，并直接使用招新题目提供的三个划分列表
（train.txt / val.txt / test.txt）。

与 LRS3 的主要差异
------------------
1. LRS2 main 的每个 clip 已经是切分好的短句（<= 约 15s），并自带逐句标注，
   因此**不需要** LRS3 那种「把长视频按时间戳切成短句」的 step 1/2。
2. LRS2 没有 pretrain 划分（题目 Dataset.md 已提示），所以本脚本只处理
   train / val / test 三个划分。
3. LRS3 用 file.list 的目录前缀区分划分；LRS2 直接用题目给的三个列表区分。

产出（供后续官方脚本使用）
--------------------------
<work_dir>/file.list   每行一个 fid，形如 train/5535415699068794046/00001
<work_dir>/label.list  与 file.list 一一对应的转写文本
<work_dir>/lrs2.prepare.summary.json  统计信息

后续步骤（官方脚本，原样使用）
------------------------------
  python count_frames.py   --root <work_dir> --manifest <work_dir>/file.list --nshard N --rank R
  python detect_landmark.py --root <work_dir> --landmark <work_dir>/landmark \
      --manifest <work_dir>/file.list --cnn_detector <dlib> --face_detector <dlib> --ffmpeg <ffmpeg>
  python align_mouth.py --video-direc <work_dir> --landmark <work_dir>/landmark \
      --filename-path <work_dir>/file.list --save-direc <work_dir>/video --mean-face <dlib> --ffmpeg <ffmpeg>
  python lrs2_manifest.py --work-dir <work_dir> --file-list <work_dir>/file.list ...

用法
----
python lrs2_prepare.py \
    --lrs2-root /path/to/lrs2/main \
    --datalist  /path/to/lrs2_datalist \
    --work-dir  /path/to/lrs2_data \
    --sed-dir   /path/to/lrs2_data/audio
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

# 划分列表文件名 -> LRS2 main 下的真实目录名
SPLITS = ("train", "val", "test")
# ★ fid 前缀必须用 **真实目录名**（train/val/test），不能改成 valid。
#   原因：官方的 detect_landmark.py 与 align_mouth.py 都会按
#       os.path.join(root, fid + '.mp4')
#   去拼路径，所以 file.list 的前缀必须和 LRS2 main 下的目录名逐字一致，
#   否则会报 "File does not exist"。
#   fairseq 需要的 valid.tsv/valid.wrd 由 lrs2_manifest.py 负责改名产出，
#   与 fid 前缀无关。


def read_list(path):
    """读取划分列表。返回 [(fid, extra_label_or_None), ...]，fid 形如 5535415699068794046/00001"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            fid = parts[0]
            # test.txt 形如：6330311066473698535/00011 NF   <- 第二列是 NF/MV 标记，不是转写
            extra = parts[1] if len(parts) > 1 else None
            if fid.endswith(".mp4"):
                fid = fid[:-4]
            items.append((fid, extra))
    return items


def find_label_file(lrs2_root, split_dir, fid):
    """在 LRS2 main 里定位某个 clip 的标注文件。

    LRS2 的目录结构大致是：
        main/<split>/<video_id>/<clip_id>.mp4
        main/<split>/<video_id>/<clip_id>.txt
    这里对几种常见结构都做尝试。
    """
    candidates = [
        os.path.join(lrs2_root, split_dir, fid + ".txt"),
        os.path.join(lrs2_root, split_dir, os.path.basename(os.path.dirname(fid)),
                     os.path.basename(fid) + ".txt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def parse_label(txt_path):
    """从 LRS2 的 .txt 标注文件里取出转写文本。

    LRS2 逐句标注文件的典型格式（前两行为元信息）：
        0 00:00:00.000 00:00:02.480
        TEXT: hello world
    也有的版本直接就是一行纯文本，这里都兼容。
    """
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]
    text = None
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        up = s.upper()
        if up.startswith("TEXT:"):
            text = s.split(":", 1)[1].strip()
            break
    if text is None:
        # 回退：跳过形如 "0 00:00:00.000 00:00:02.480" 的元信息行，取剩下的第一行
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            toks = s.split()
            if len(toks) >= 3 and ":" in toks[1]:
                continue
            text = s
            break
    if text is None:
        raise ValueError("无法从 %s 解析转写文本" % txt_path)
    return " ".join(text.split())


def main():
    ap = argparse.ArgumentParser(
        description="AV-HuBERT LRS2(main) 数据准备：生成 file.list / label.list",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--lrs2-root", required=True,
                    help="LRS2 main 数据集根目录（其下应有 train/ val/ test/ 三个子目录）")
    ap.add_argument("--datalist", required=True,
                    help="招新题目提供的划分列表目录（含 train.txt/val.txt/test.txt）")
    ap.add_argument("--work-dir", required=True,
                    help="输出工作目录（file.list/label.list/audio/video/landmark 都放这里）")
    ap.add_argument("--copy-video-to", default=None,
                    help="可选：把原始 mp4 按 fid 结构复制/硬链接到该目录（默认不复制，直接用原路径）")
    ap.add_argument("--link", action="store_true",
                    help="配合 --copy-video-to 使用硬链接而非复制（同一文件系统时省空间）")
    ap.add_argument("--split-prefix", action="store_true", default=True,
                    help="fid 前加划分前缀（train/val/test，与 LRS2 真实目录名一致），默认开启")
    ap.add_argument("--no-split-prefix", dest="split_prefix", action="store_false")
    ap.add_argument("--strict", action="store_true", default=True,
                    help="只要有 clip 缺标注就报错退出（默认开启）")
    ap.add_argument("--allow-missing", dest="strict", action="store_false",
                    help="允许跳过缺标注的 clip（会打印警告）")
    args = ap.parse_args()

    # ---------- 检查输入 ----------
    if not os.path.isdir(args.lrs2_root):
        sys.exit("错误：--lrs2-root 不存在：%s" % args.lrs2_root)
    if not os.path.isdir(args.datalist):
        sys.exit("错误：--datalist 不存在：%s" % args.datalist)

    os.makedirs(args.work_dir, exist_ok=True)

    fids_all, labels_all = [], []
    stats = OrderedDict()

    for split in SPLITS:
        list_fn = os.path.join(args.datalist, "%s.txt" % split)
        if not os.path.isfile(list_fn):
            sys.exit("错误：找不到划分列表 %s" % list_fn)

        items = read_list(list_fn)
        out_prefix = split if args.split_prefix else ""

        n_ok, n_missing_label, n_missing_media, missing_examples = 0, 0, 0, []
        part_fids, part_labels = [], []

        for fid, _extra in items:
            txt = find_label_file(args.lrs2_root, split, fid)
            if txt is None:
                n_missing_label += 1
                if len(missing_examples) < 5:
                    missing_examples.append(fid)
                if args.strict:
                    continue
                else:
                    continue
            try:
                label = parse_label(txt)
            except Exception as e:
                n_missing_label += 1
                if len(missing_examples) < 5:
                    missing_examples.append("%s (%s)" % (fid, e))
                continue

            media = os.path.join(args.lrs2_root, split, fid + ".mp4")
            if not os.path.isfile(media):
                n_missing_media += 1
                if len(missing_examples) < 5:
                    missing_examples.append(fid + " [mp4 missing]")
                continue

            out_fid = ("%s/%s" % (out_prefix, fid)) if out_prefix else fid
            part_fids.append(out_fid)
            part_labels.append(label)
            n_ok += 1

        stats[split] = OrderedDict(
            listed=len(items), used=n_ok,
            missing_label=n_missing_label, missing_mp4=n_missing_media,
            examples=missing_examples,
        )
        print("[%s] 列表中 %d 条，成功 %d 条，缺标注 %d，缺 mp4 %d"
              % (split, len(items), n_ok, n_missing_label, n_missing_media))
        if missing_examples:
            print("        示例: %s" % ", ".join(str(x) for x in missing_examples[:5]))

        fids_all.extend(part_fids)
        labels_all.extend(part_labels)

    # ---------- 严格模式：有任何缺失就退出 ----------
    total_missing = sum(v["missing_label"] + v["missing_mp4"] for v in stats.values())
    if args.strict and total_missing > 0:
        print()
        print("=" * 72)
        print("错误：共有 %d 条 clip 找不到标注或 mp4。这通常说明：" % total_missing)
        print("  1) 数据集没有下载完整（LRS2 main 应包含 train/val/test 三个子目录）；")
        print("  2) --lrs2-root 指错了层级（应指向 main 目录，而不是它的上级）；")
        print("  3) 列表里的 clip 与本地数据版本不一致。")
        print("确认无误后，可加 --allow-missing 跳过缺失项继续。")
        print("=" * 72)
        sys.exit(1)

    # ---------- 落盘 ----------
    file_list = os.path.join(args.work_dir, "file.list")
    label_list = os.path.join(args.work_dir, "label.list")
    with open(file_list, "w", encoding="utf-8") as fo:
        fo.write("\n".join(fids_all) + "\n")
    with open(label_list, "w", encoding="utf-8") as fo:
        fo.write("\n".join(labels_all) + "\n")

    summary = OrderedDict(
        lrs2_root=os.path.abspath(args.lrs2_root),
        datalist=os.path.abspath(args.datalist),
        work_dir=os.path.abspath(args.work_dir),
        split_prefix=args.split_prefix,
        total=len(fids_all),
        per_split=stats,
    )
    with open(os.path.join(args.work_dir, "lrs2.prepare.summary.json"), "w", encoding="utf-8") as fo:
        json.dump(summary, fo, ensure_ascii=False, indent=2)

    print()
    print("完成：共 %d 条" % len(fids_all))
    print("  fid  列表 -> %s" % file_list)
    print("  文本 列表 -> %s" % label_list)
    print("  统计信息  -> %s" % os.path.join(args.work_dir, "lrs2.prepare.summary.json"))
    print()
    print("下一步：")
    print("  1) 抽音频 + 唇部 ROI（官方脚本，见 pipeline_lrs2.sh 的 step 2/3）")
    print("  2) 数帧数：  python count_frames.py --root %s --manifest %s --nshard 8 --rank 0"
          % (args.work_dir, file_list))
    print("  3) 生成 manifest： python lrs2_manifest.py --work-dir %s" % args.work_dir)


if __name__ == "__main__":
    main()
