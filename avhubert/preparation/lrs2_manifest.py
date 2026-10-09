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
import re
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
    # 坑：小样本验证（LIMIT=30）时 sentencepiece 会自动降级到很小的 vocab，
    # 但那批文件仍然叫 spm_unigram1000.*，并会写下 vocab_size.txt。之后在同一个
    # WORK 目录里跑全量时，下面这个"已存在就跳过"会把**只有几十个 piece 的降级词表**
    # 直接拿去用，而 vocab_size.txt 此前没有任何地方读，于是全程静默降质。
    # 这里改成：记录的词表大小与本次要求不一致就重训（并说清原因）。
    prev_size = None
    prev_size_fn = Path(work_dir) / "vocab_size.txt"
    if prev_size_fn.is_file():
        try:
            prev_size = int(prev_size_fn.read_text().strip())
        except Exception:
            prev_size = None
    if prev_size is not None and prev_size != args.vocab_size \
            and (vocab_dir / ("spm_unigram%d.txt" % args.vocab_size)).is_file():
        print("警告：已有词表是降级产物（记录 vocab_size=%d，本次要求 %d）。"
              % (prev_size, args.vocab_size))
        print("      将删除并重新训练，避免把降级词表用到全量实验上。")
        for suffix in (".model", ".vocab", ".txt"):
            p = Path(spm_prefix.as_posix() + suffix)
            if p.is_file():
                p.unlink()
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
            # 正常路径也要记录实际词表大小：否则"降级过、又重训成 1000"时
            # vocab_size.txt 会停留在旧值，上面的校验就会误判。
            with open(os.path.join(work_dir, "vocab_size.txt"), "w") as fo:
                fo.write("%d\n" % args.vocab_size)
        except RuntimeError as e:
            # sentencepiece 在语料过小时会报
            #   Vocabulary size too high (N). Please set it to a value <= M.
            # 这只在小样本验证（如 LIMIT=30）时出现；全量语料不会有此问题。
            # 这里按报错给出的上限自动降级重试，避免把"样本太小"误判为流程失败。
            m = re.search(r"value <= (\d+)", str(e))
            if not m:
                raise
            fallback = int(m.group(1))
            print("警告：vocab_size=%d 超过当前语料可支持的上限 %d（通常因为样本量太小）。"
                  % (args.vocab_size, fallback))
            print("      自动降级为 vocab_size=%d 重试。正式全量运行时请保持 1000。" % fallback)
            # 清理可能残留的半成品
            for suffix in (".model", ".vocab", ".txt"):
                p = Path(spm_prefix.as_posix() + suffix)
                if p.is_file():
                    p.unlink()
            gen_vocab(Path(tmp_txt), spm_prefix, "unigram", fallback)
            # 记录实际使用的词表大小，供后续步骤引用
            with open(os.path.join(work_dir, "vocab_size.txt"), "w") as fo:
                fo.write("%d\n" % fallback)
        finally:
            os.unlink(tmp_txt)
    vocab_path = spm_prefix.as_posix() + ".txt"
    if not os.path.isfile(vocab_path):
        sys.exit("错误：词表生成失败，未见 %s" % vocab_path)

    # ---------- 划分：优先用 split.list，回退到 fid 前缀 ----------
    #   flat  布局：fid = <video_id>/<clip_id>，目录里没有划分信息，
    #              必须用 lrs2_prepare.py 产出的 split.list
    #   split 布局：fid = <split>/<video_id>/<clip_id>，可直接读前缀
    buckets = {"train": [], "val": [], "test": []}
    split_list_fn = os.path.join(work_dir, "split.list")
    unknown = 0
    if os.path.isfile(split_list_fn):
        splits = read_lines(split_list_fn)
        if len(splits) != len(fids):
            sys.exit("错误：split.list(%d) 与 file.list(%d) 长度不一致"
                     % (len(splits), len(fids)))
        for i, s in enumerate(splits):
            if s in buckets:
                buckets[s].append(i)
            else:
                unknown += 1
        print("划分来源：split.list（flat 布局）")
    else:
        for i, fid in enumerate(fids):
            prefix = fid.split("/")[0]
            if prefix in buckets:
                buckets[prefix].append(i)
            else:
                unknown += 1
        print("划分来源：fid 前缀（split 布局；如需 flat 请先产出 split.list）")
    if unknown:
        print("警告：%d 条的划分无法识别" % unknown)

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
            # tags 的键是 datalist 里的原始 fid（形如 <video_id>/<clip_id>）；
            # flat 布局下 file.list 的 fid 与之一致，split 布局下需去掉前缀。
            def norm(fid):
                parts = fid.split("/")
                return "/".join(parts[-2:]) if len(parts) > 2 else fid

            mv_idx = [i for i in buckets["test"]
                      if tags.get(norm(fids[i]).rsplit(".", 1)[0], "") == "MV"]
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
