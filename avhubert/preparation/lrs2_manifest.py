#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lrs2_manifest.py -- 生成 AV-HuBERT fine-tune 所需的 LRS2 数据目录

对应官方的 avhubert/preparation/lrs3_manifest.py，但适配 LRS2：
  * 划分直接来自 lrs2_prepare.py 生成的 file.list 里的前缀（train/val/test）；
  * 输出文件名做了一次改名：val 划分写成 valid.tsv / valid.wrd，
    以匹配 fairseq 配置中的 dataset.valid_subset: valid；
  * LRS2 main 没有 pretrain 划分，因此不再区分 30h / 433h 训练集，
    但仍支持 --max-train 只取一部分做小规模冒烟测试。

产出目录结构（<out-dir>）：
    train.tsv    valid.tsv    test.tsv        # manifest，首行是 root，其余每行 5 列
    train.wrd    valid.wrd    test.wrd        # 逐行转写
    dict.wrd.txt                              # fairseq 词典（= sentencepiece 词表导出的 .txt）
    spm<vocab_size>/spm_unigram<vocab_size>.{model,vocab,txt}
    test_mv.wrd（可选）                        # 仅 mouth-visible 子集，用于分口径汇报

tsv 每列含义（与 hubert_dataset.load_audio_visual 一致）：
    id <TAB> video_path <TAB> audio_path <TAB> n_frames_video <TAB> n_frames_audio
    —— 注意 items[-2] 被当作帧数，因此这两列的顺序是「视频帧数, 音频帧数」。

用法
----
python lrs2_manifest.py --work-dir /path/to/lrs2_data --datalist /path/to/lrs2_datalist
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_subword import gen_vocab  # noqa: E402  官方词表生成器

# fid 前缀（= LRS2 真实目录名）与输出文件名（= fairseq 的 subset 名）的映射。
# fairseq 配置里写的是 dataset.valid_subset: valid，任务会去找 {data}/valid.tsv，
# 而 LRS2 的目录名是 val，所以这里必须改名。
BUCKET_TO_FILE = {"train": "train", "val": "valid", "test": "test"}


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="生成 AV-HuBERT fine-tune 用的 LRS2 数据目录",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--work-dir", required=True,
                    help="lrs2_prepare.py 的输出目录（含 file.list/label.list/audio/video）")
    ap.add_argument("--out-dir", default=None,
                    help="输出数据目录，默认 <work-dir>/data")
    ap.add_argument("--datalist", default=None,
                    help="招新题目给的划分列表目录；给出后会额外生成 test_mv.wrd 等分口径标签")
    ap.add_argument("--vocab-size", type=int, default=1000,
                    help="sentencepiece 词表大小（官方 LRS3 配置用 1000）")
    ap.add_argument("--max-train", type=int, default=None,
                    help="仅取前 N 条训练样本，用于小规模冒烟测试")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    work_dir = args.work_dir
    out_dir = args.out_dir or os.path.join(work_dir, "data")

    file_list = os.path.join(work_dir, "file.list")
    label_list = os.path.join(work_dir, "label.list")
    nframes_audio_fn = os.path.join(work_dir, "nframes.audio")
    nframes_video_fn = os.path.join(work_dir, "nframes.video")

    for fn, hint in [
        (file_list, "先运行 lrs2_prepare.py"),
        (label_list, "先运行 lrs2_prepare.py"),
        (nframes_audio_fn, "先运行 count_frames.py 并合并分片"),
        (nframes_video_fn, "先运行 count_frames.py 并合并分片"),
    ]:
        if not os.path.isfile(fn):
            sys.exit("错误：找不到 %s —— %s" % (fn, hint))

    fids = read_lines(file_list)
    labels = [x.lower() for x in read_lines(label_list)]
    nf_audio = read_lines(nframes_audio_fn)
    nf_video = read_lines(nframes_video_fn)

    if not (len(fids) == len(labels) == len(nf_audio) == len(nf_video)):
        sys.exit("错误：file.list(%d) / label.list(%d) / nframes.audio(%d) / nframes.video(%d) 长度不一致"
                 % (len(fids), len(labels), len(nf_audio), len(nf_video)))

    audio_dir = os.path.join(work_dir, "audio")
    video_dir = os.path.join(work_dir, "video")

    # ---------- 词表 ----------
    vocab_dir = Path(work_dir) / ("spm%d" % args.vocab_size)
    vocab_dir.mkdir(exist_ok=True)
    spm_prefix = vocab_dir / ("spm_unigram%d" % args.vocab_size)
    if (vocab_dir / ("spm_unigram%d.txt" % args.vocab_size)).is_file():
        print("词表已存在，跳过 sentencepiece 训练：%s" % spm_prefix.as_posix() + ".txt")
    else:
        # sentencepiece 训练集：官方用全部 label.list（含 test，属官方行为，保持一致）
        import tempfile
        print("正在训练 sentencepiece unigram（vocab_size=%d）..." % args.vocab_size)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            for t in labels:
                f.write(t + "\n")
            tmp_txt = f.name
        try:
            gen_vocab(Path(tmp_txt), spm_prefix, "unigram", args.vocab_size)
        finally:
            os.unlink(tmp_txt)
    vocab_path = spm_prefix.as_posix() + ".txt"
    if not os.path.isfile(vocab_path):
        sys.exit("错误：词表生成失败，未见 %s" % vocab_path)

    # ---------- 按前缀划分 ----------
    buckets = {"train": [], "val": [], "test": []}
    unknown = 0
    for i, fid in enumerate(fids):
        prefix = fid.split("/")[0]
        if prefix in buckets:
            buckets[prefix].append(i)
        else:
            unknown += 1
    if unknown:
        print("警告：%d 条 fid 的划分前缀无法识别（既不是 train/ 也不是 val//test/）" % unknown)

    if args.max_train is not None and len(buckets["train"]) > args.max_train:
        import random
        random.Random(args.seed).shuffle(buckets["train"])
        buckets["train"] = sorted(buckets["train"][:args.max_train])

    os.makedirs(out_dir, exist_ok=True)

    def write_split(name, indices):
        tsv_path = os.path.join(out_dir, "%s.tsv" % name)
        wrd_path = os.path.join(out_dir, "%s.wrd" % name)
        with open(tsv_path, "w", encoding="utf-8") as fo:
            fo.write("/\n")  # 首行是 root（load_audio_visual 会跳过它）
            for i in indices:
                fid = fids[i]
                fo.write("\t".join([
                    fid,
                    os.path.abspath(os.path.join(video_dir, fid + ".mp4")),
                    os.path.abspath(os.path.join(audio_dir, fid + ".wav")),
                    nf_video[i],   # items[-2] -> 视频帧数
                    nf_audio[i],   # items[-1] -> 音频帧数
                ]) + "\n")
        with open(wrd_path, "w", encoding="utf-8") as fo:
            for i in indices:
                fo.write("%s\n" % labels[i])
        dur_h = sum(int(nf_video[i]) for i in indices) / 25.0 / 3600.0
        print("  %-6s %6d 条  -> %s  (约 %.2f 小时)" % (name, len(indices), tsv_path, dur_h))
        return len(indices)

    print("生成数据目录 %s" % out_dir)
    for bucket, fname in BUCKET_TO_FILE.items():
        write_split(fname, buckets[bucket])

    # ---------- 词典 ----------
    dict_path = os.path.join(out_dir, "dict.wrd.txt")
    shutil.copyfile(vocab_path, dict_path)
    print("  词典   -> %s" % dict_path)

    # ---------- 可选：test 的 MV 子集（分口径汇报用） ----------
    if args.datalist:
        tag_fn = os.path.join(args.datalist, "test.txt")
        if os.path.isfile(tag_fn):
            tags = {}
            with open(tag_fn, "r", encoding="utf-8") as f:
                for ln in f:
                    p = ln.split()
                    if len(p) >= 2:
                        tags[p[0].rstrip("/")] = p[1].upper()
            mv_idx = [i for i in buckets["test"]
                      if tags.get(fids[i].split("/", 1)[1].rsplit(".", 1)[0], "") == "MV"]
            if mv_idx:
                with open(os.path.join(out_dir, "test_mv.wrd"), "w", encoding="utf-8") as fo:
                    for i in mv_idx:
                        fo.write("%s\n" % labels[i])
                with open(os.path.join(out_dir, "test_mv.tsv"), "w", encoding="utf-8") as fo:
                    fo.write("/\n")
                    for i in mv_idx:
                        fid = fids[i]
                        fo.write("\t".join([
                            fid,
                            os.path.abspath(os.path.join(video_dir, fid + ".mp4")),
                            os.path.abspath(os.path.join(audio_dir, fid + ".wav")),
                            nf_video[i], nf_audio[i],
                        ]) + "\n")
                print("  MV 子集 %5d 条  -> %s" % (len(mv_idx), os.path.join(out_dir, "test_mv.tsv")))

    print()
    print("完成。fine-tune 时把 task.data / task.label_dir 指向：%s" % os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
