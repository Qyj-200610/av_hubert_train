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

# 划分列表文件名
SPLITS = ("train", "val", "test")

# ---------------------------------------------------------------------------
# 两种数据布局（实测：本考核从 AI Studio 下载的 main 是 "flat"）
#
#   flat  : main/<video_id>/<clip_id>.mp4          （共 3783 个 video_id 目录）
#           实测与 datalist 完全对应：视频 ID 就是顶层目录名。
#   split : main/<split>/<video_id>/<clip_id>.mp4  （部分 LRS2 打包方式）
#
# 为什么要做成选项：官方的 detect_landmark.py / align_mouth.py 都按
#     os.path.join(root, fid + '.mp4')
# 拼路径，所以 file.list 里的 fid 必须与实际目录结构逐字一致，否则会报
# "File does not exist"。这里按布局生成对应的 fid。
#   flat  -> fid = "<video_id>/<clip_id>"
#   split -> fid = "<split>/<video_id>/<clip_id>"
# ---------------------------------------------------------------------------


def build_vid2split(datalist_dir):
    """从三个划分列表建立 video_id -> split 的映射。

    flat 布局下目录里没有划分信息，划分归属只能由 datalist 决定。
    """
    vid2split = {}
    for split in SPLITS:
        fn = os.path.join(datalist_dir, "%s.txt" % split)
        if not os.path.isfile(fn):
            continue
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                vid = parts[0].split("/")[0]
                if vid in vid2split and vid2split[vid] != split:
                    # 同一个 video_id 跨划分出现（LRS2 里确实可能），标为冲突
                    vid2split[vid] = "__conflict__"
                else:
                    vid2split.setdefault(vid, split)
    return vid2split


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


def label_file_of(lrs2_root, layout, split, fid):
    """按布局返回 clip 标注文件（<clip>.txt）的路径。

    - flat : main/<fid>.txt，其中 fid = <video_id>/<clip_id>
    - split: main/<split>/<fid>.txt
    两种布局下 <clip>.txt 都与 <clip>.mp4 同目录同名，故直接替换后缀。
    """
    if layout == "split":
        base = os.path.join(lrs2_root, split, fid)
    else:
        base = os.path.join(lrs2_root, fid)
    return base + ".txt"


def media_file_of(lrs2_root, layout, split, fid):
    """按布局返回 clip 视频文件（<clip>.mp4）的路径。"""
    if layout == "split":
        base = os.path.join(lrs2_root, split, fid)
    else:
        base = os.path.join(lrs2_root, fid)
    return base + ".mp4"


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
                    help="LRS2 main 数据集根目录（应指向 main 目录本身）")
    ap.add_argument("--datalist", required=True,
                    help="招新题目提供的划分列表目录（含 train.txt/val.txt/test.txt）")
    ap.add_argument("--work-dir", required=True,
                    help="输出工作目录（file.list/label.list/audio/video/landmark 都放这里）")
    ap.add_argument("--layout", choices=("flat", "split"), default="flat",
                    help="目录布局。flat: main/<video_id>/<clip>.mp4（本考核实测即为该布局）；"
                         "split: main/<split>/<video_id>/<clip>.mp4")
    ap.add_argument("--copy-video-to", default=None,
                    help="（未实现）原文档说把原始 mp4 复制/硬链接到该目录；实际上这个参数"
                         "此前被静默忽略，现在改为显式报错，避免「以为复制了其实没有」")
    ap.add_argument("--link", action="store_true",
                    help="（未实现）配合 --copy-video-to 的硬链接开关，同上")
    ap.add_argument("--split-prefix", action="store_true", default=True,
                    help="仅 split 布局生效：fid 前加划分前缀（train/val/test）")
    ap.add_argument("--no-split-prefix", dest="split_prefix", action="store_false")
    ap.add_argument("--strict", action="store_true", default=True,
                    help="只要有 clip 缺标注就报错退出（默认开启）")
    ap.add_argument("--allow-missing", dest="strict", action="store_false",
                    help="允许跳过缺标注的 clip（会打印警告）")
    # 小样本验证用：只取每个划分的前 N 条
    ap.add_argument("--max-train", type=int, default=None,
                    help="train 划分最多取多少条（用于小样本验证全流程）")
    ap.add_argument("--max-valid", type=int, default=None,
                    help="val 划分最多取多少条")
    ap.add_argument("--max-test", type=int, default=None,
                    help="test 划分最多取多少条")
    args = ap.parse_args()

    limit_map = {"train": args.max_train, "val": args.max_valid, "test": args.max_test}

    # ---------- 检查输入 ----------
    # --copy-video-to / --link 从来没有被实现过（参数被解析后无人使用），
    # 传了会"静默什么都不做"。改成明确报错，免得以为 mp4 已经固化下来了。
    # （本流程一直是直接用 --lrs2-root 下的原始 mp4：detect_landmark/align_mouth
    #   的 --root / --video-direc 都指向那里。）
    if args.copy_video_to or args.link:
        sys.exit("错误：--copy-video-to / --link 尚未实现（此前会被静默忽略）。\n"
                 "      本流程直接使用 --lrs2-root 下的原始 mp4；若需要把 mp4 固化到\n"
                 "      工作目录，请自行 cp/rsync 后再把 --lrs2-root 指过去。")
    if not os.path.isdir(args.lrs2_root):
        sys.exit("错误：--lrs2-root 不存在：%s" % args.lrs2_root)
    if not os.path.isdir(args.datalist):
        sys.exit("错误：--datalist 不存在：%s" % args.datalist)

    os.makedirs(args.work_dir, exist_ok=True)

    # flat 布局下目录里没有划分信息，先建立 video_id -> split 映射
    vid2split = build_vid2split(args.datalist) if args.layout == "flat" else {}
    if args.layout == "flat":
        n_conf = sum(1 for v in vid2split.values() if v == "__conflict__")
        print("布局=flat，从 datalist 建立 video_id->split 映射：%d 个 video_id%s"
              % (len(vid2split), ("，其中 %d 个跨划分冲突" % n_conf) if n_conf else ""))

    fids_all, labels_all, splits_all = [], [], []
    stats = OrderedDict()

    for split in SPLITS:
        list_fn = os.path.join(args.datalist, "%s.txt" % split)
        if not os.path.isfile(list_fn):
            sys.exit("错误：找不到划分列表 %s" % list_fn)

        items = read_list(list_fn)
        # 小样本验证：只保留前 N 条
        lim = limit_map.get(split)
        if lim is not None and lim > 0 and len(items) > lim:
            print("[%s] 受 --max-%s=%d 限制，从 %d 条中取前 %d 条"
                  % (split, "valid" if split == "val" else split, lim, len(items), lim))
            items = items[:lim]

        n_ok, n_missing_label, n_missing_media, missing_examples = 0, 0, 0, []
        part_fids, part_labels = [], []

        for fid, _extra in items:
            txt = label_file_of(args.lrs2_root, args.layout, split, fid)
            if not os.path.isfile(txt):
                n_missing_label += 1
                if len(missing_examples) < 5:
                    missing_examples.append(fid)
                continue
            try:
                label = parse_label(txt)
            except Exception as e:
                n_missing_label += 1
                if len(missing_examples) < 5:
                    missing_examples.append("%s (%s)" % (fid, e))
                continue

            media = media_file_of(args.lrs2_root, args.layout, split, fid)
            if not os.path.isfile(media):
                n_missing_media += 1
                if len(missing_examples) < 5:
                    missing_examples.append(fid + " [mp4 missing]")
                continue

            # flat 布局下 fid 就是 <video_id>/<clip_id>，split 归属另行记录
            out_fid = ("%s/%s" % (split, fid)) if (args.layout == "split" and args.split_prefix) else fid
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
        splits_all.extend([split] * len(part_fids))

    # ---------- 严格模式：有任何缺失就退出 ----------
    total_missing = sum(v["missing_label"] + v["missing_mp4"] for v in stats.values())
    if args.strict and total_missing > 0:
        print()
        print("=" * 72)
        print("错误：共有 %d 条 clip 找不到标注或 mp4。这通常说明：" % total_missing)
        print("  1) 数据集还没下载完整（本考核的 main 约 48164 个 clip）；")
        print("  2) --lrs2-root 指错了层级（应指向 main 目录本身）；")
        print("  3) 布局判断错了：本数据集实测是 flat（main/<video_id>/<clip>.mp4），")
        print("     若你的数据是 main/<split>/<video_id>/... 请加 --layout split；")
        print("  4) 列表里的 clip 与本地数据版本不一致。")
        print("确认无误后，可加 --allow-missing 跳过缺失项继续。")
        print("=" * 72)
        sys.exit(1)

    # ---------- 重复 / 跨划分泄漏检查（原实现只把冲突数打印出来，不做任何拦截）----------
    # 为什么必须有：flat 布局下 out_fid 就是 fid，所以同一个 fid 若同时出现在两份列表里，
    # 会被**分别写进 train.tsv 与 test.tsv，且指向同一个 video/<fid>.mp4**
    # —— 测试片段静默进了训练集，WER 会被抬高，而日志里一个字都不会提。
    # （已核实本次题目给的三个列表是干净的：0 重复、两两交集为空、无共享 video_id；
    #   这个检查是为了防止以后换数据版本时踩坑。）
    dup_all = len(fids_all) - len(set(fids_all))
    if dup_all:
        from collections import Counter
        dup_items = [x for x, c in Counter(fids_all).items() if c > 1]
        print()
        print("=" * 72)
        print("错误：共 %d 条 clip 在不同划分里重复出现（重复条目 %d 个）。" % (dup_all, len(dup_items)))
        print("      同一 clip 会被写进多份 tsv，测试集数据会泄漏进训练集。")
        print("      示例：%s" % ", ".join(dup_items[:5]))
        print("      请检查 --datalist 里的三个列表是否有重叠。")
        print("=" * 72)
        sys.exit(1)
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        sa = {f for f, s in zip(fids_all, splits_all) if s == a}
        sb = {f for f, s in zip(fids_all, splits_all) if s == b}
        inter = sa & sb
        if inter:
            print("=" * 72)
            print("错误：%s 与 %s 划分有 %d 条重叠（示例：%s）" % (a, b, len(inter), list(inter)[:3]))
            print("=" * 72)
            sys.exit(1)

    # ---------- 落盘 ----------
    # 三个列表都用「临时文件 + os.replace」原子发布。
    # 原因：file.list 是**整个流水线的规模基准**——`pipeline_lrs2.sh` 用
    # `need=$(wc -l < file.list)` 判断"产物齐了没有"，分片切分也按它的长度算。
    # 如果这一步被杀/磁盘满，留下一个**短了**的 file.list，那么"齐备检查"会
    # 按更小的数字通过，后续步骤只处理这一部分，最后训练集静默变小——
    # 而 verify_dataset.py 校验的是"三者自洽"，也发现不了。
    file_list = os.path.join(args.work_dir, "file.list")
    label_list = os.path.join(args.work_dir, "label.list")
    # 额外产出 split.list：flat 布局下 fid 不带划分前缀，
    # lrs2_manifest.py 需要靠它来划分 train/valid/test。
    split_list = os.path.join(args.work_dir, "split.list")

    def _atomic_write(path, text):
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as fo:
                fo.write(text)
                fo.flush()
                os.fsync(fo.fileno())
            os.replace(tmp, path)      # 要么没有，要么完整
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise

    _atomic_write(file_list, "\n".join(fids_all) + "\n")
    _atomic_write(label_list, "\n".join(labels_all) + "\n")
    _atomic_write(split_list, "\n".join(splits_all) + "\n")

    # 写盘后立刻自检三个列表必须等长（等长是下游所有对齐检查的前提）
    _n = [len(fids_all), len(labels_all), len(splits_all)]
    if len(set(_n)) != 1:
        print("错误：file.list / label.list / split.list 长度不一致：%s" % _n)
        sys.exit(1)

    summary = OrderedDict(
        lrs2_root=os.path.abspath(args.lrs2_root),
        datalist=os.path.abspath(args.datalist),
        work_dir=os.path.abspath(args.work_dir),
        layout=args.layout,
        split_prefix=args.split_prefix,
        total=len(fids_all),
        per_split=stats,
    )
    _atomic_write(os.path.join(args.work_dir, "lrs2.prepare.summary.json"),
                  json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print()
    print("完成：共 %d 条" % len(fids_all))
    print("  fid  列表 -> %s" % file_list)
    print("  文本 列表 -> %s" % label_list)
    print("  划分 列表 -> %s" % split_list)
    print("  统计信息  -> %s" % os.path.join(args.work_dir, "lrs2.prepare.summary.json"))
    print()
    print("fid 样例：%s" % (fids_all[0] if fids_all else "(空)"))
    print()
    print("下一步：")
    print("  1) 抽音频 + 唇部 ROI（官方脚本，见 pipeline_lrs2.sh 的 step 2/3）")
    print("  2) 数帧数：  python count_frames.py --root %s --manifest %s --nshard 8 --rank 0"
          % (args.work_dir, file_list))
    print("  3) 生成 manifest： python lrs2_manifest.py --work-dir %s" % args.work_dir)


if __name__ == "__main__":
    main()
