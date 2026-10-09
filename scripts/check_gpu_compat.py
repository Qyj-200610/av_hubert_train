#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_gpu_compat.py -- 租到云 GPU 后的第一件事：确认这张卡能用本项目的 torch

背景
----
fairseq 被 pin 在 2021-06 的 commit，配套 torch 1.13.1+cu117。
torch 的预编译包里有一份**固定的算力（compute capability）列表**，
显卡算力不在列表里，kernel 就跑不起来——与「CUDA 版本够不够新」是两回事。

本脚本做三件事：
  1. 打印本机 GPU、驱动、CUDA、算力，以及当前 torch 的架构列表
  2. 判断算力是否被支持，并给出明确的「能用 / 不能用」结论
  3. 给出建议：若不可用，告诉你该换什么卡

用法（在云机器上，装好环境后执行）
----------------------------------
    python scripts/check_gpu_compat.py

建议流程：租机 -> 建环境 -> **跑本脚本** -> 通过了再下载数据，别浪费机时。
"""

import sys

# Windows 控制台默认是 GBK，输出中文/符号会抛 UnicodeEncodeError。
# 重设为 UTF-8 可避免；在 Linux（云机器）上这是无害的空操作。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

OK = '\033[92m'
BAD = '\033[91m'
WARN = '\033[93m'
DIM = '\033[90m'
END = '\033[0m'

# 输出被重定向到文件时关掉颜色，避免出现乱码转义符
if not sys.stdout.isatty():
    OK = BAD = WARN = DIM = END = ''

# 各类卡的算力对照（用于给出换卡建议）
KNOWN_GPUS = [
    ('Tesla V100', 'sm_70'),
    ('RTX 2080 Ti', 'sm_75'),
    ('A100', 'sm_80'),
    ('RTX 3090', 'sm_86'),
    ('RTX 4090', 'sm_89'),
    ('H100', 'sm_90'),
    ('B100 / B200', 'sm_100'),      # Blackwell 数据中心卡是 sm_100，不是 sm_120
    ('RTX 50 系 (Blackwell)', 'sm_120'),
]
# torch 1.13.1+cu117 的典型架构列表
TORCH_113_ARCHS = ['sm_37', 'sm_50', 'sm_60', 'sm_61', 'sm_70', 'sm_75', 'sm_80', 'sm_86']


def main():
    print('=' * 70)
    print('AV-HuBERT (LRS2) 云 GPU 兼容性检查')
    print('=' * 70)

    try:
        import torch
    except ImportError:
        print(BAD + '✗ 无法 import torch' + END)
        print('  请先创建并激活环境：')
        print('    conda env create -f environment.yml && conda activate avhubert')
        return 2

    print()
    print('--- 环境 ---')
    print('  python   :', sys.version.split()[0])
    print('  torch    :', torch.__version__)
    print('  cuda(运行时) :', torch.version.cuda)

    if sys.version_info >= (3, 11):
        print(WARN + '  ! Python >= 3.11：fairseq 会被 import 时的 dataclass 检查拒绝，' + END)
        print(WARN + '    请用 environment.yml 里的 python=3.8' + END)

    print()
    print('--- 当前 torch 构建的架构列表 ---')
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception as e:
        print(BAD + '  ✗ 读取失败: %s' % e + END)
        return 2
    print('  ', arch_list)

    print()
    print('--- GPU ---')
    if not torch.cuda.is_available():
        print(BAD + '  ✗ torch.cuda.is_available() = False' + END)
        print('  常见原因：驱动太旧 / 容器没挂载 GPU / 装成了 CPU 版 torch')
        print('  排查：nvidia-smi 能否正常输出；torch 版本是否带 +cu 后缀')
        return 2

    n = torch.cuda.device_count()
    print('  可见 GPU 数 :', n)
    for i in range(n):
        print('    [%d] %s  (显存 %.1f GB)'
              % (i, torch.cuda.get_device_name(i),
                 torch.cuda.get_device_properties(i).total_memory / 1024 ** 3))

    # 逐张卡判：异构机器上只看 0 号卡会误判（例如 0 号是 V100、1 号是新卡）
    print()
    print('  逐卡算力是否在 torch 架构列表内：')
    caps = {}
    for i in range(n):
        c = torch.cuda.get_device_capability(i)
        s = 'sm_%d%d' % c
        caps[i] = s
        mark = OK + '✓' + END if s in arch_list else BAD + '✗' + END
        print('    [%d] %-28s %-7s %s' % (i, torch.cuda.get_device_name(i)[:28], s, mark))

    cap = torch.cuda.get_device_capability(0)
    sm = 'sm_%d%d' % cap
    print()
    print('  0 号卡算力 :', cap, '->', sm)

    # ---------------- 结论 ----------------
    print()
    print('=' * 70)
    # 只要**每一张**可见卡都受支持才算通过（训练脚本默认用 CUDA_VISIBLE_DEVICES 选卡）
    unsupported = sorted({s for s in caps.values() if s not in arch_list})
    supported = not unsupported

    if supported:
        print(OK + '✓ 通过：全部 %d 张卡的算力 %s 都在 torch 的架构列表内，可以开始跑实验。'
              % (n, ', '.join(sorted(set(caps.values())))) + END)
        print()
        print('  下一步：')
        print('    export LRS2_ROOT=... DATALIST=... WORK=... FFMPEG=$(which ffmpeg) DLIB=...')
        print('    bash scripts/fetch_dlib_models.sh "$DLIB"')
        print('    bash scripts/pipeline_lrs2.sh prepare   # 先验证数据列表能对上')
        return 0

    print(BAD + '✗ 不通过：算力 %s 不在 torch 的架构列表内，kernel 无法执行。'
          % ', '.join(unsupported) + END)
    print()
    print('  这不是环境装错，而是「卡太新、torch 太老」。换卡，不要往下跑。')
    print()
    print(DIM + '  本卡算力对照与建议：' + END)
    for name, smn in KNOWN_GPUS:
        if smn in unsupported:
            mark = '  <- 就是这张'
        else:
            mark = ''
        if smn in TORCH_113_ARCHS:
            verdict = OK + '可用' + END
        else:
            verdict = BAD + '不可用' + END
        print('    %-24s %-8s %s%s' % (name, smn, verdict, mark))
    print()
    print('  建议租用（算力在 sm_86 以内）：')
    print('    * RTX 3090  (sm_86, 24GB) —— 性价比最高')
    print('    * A100      (sm_80, 40/80GB) —— 最稳')
    print('    * RTX 2080 Ti / V100 —— 更老但也能用')
    print()
    print('  另一种思路（需自行验证，本项目未测）：把 torch 换成较新版本')
    print('  （如 2.x + cu121/cu124），使其架构列表覆盖本卡。')
    print('  代价是环境不再是官方验证过的组合，可能撞上 fairseq 与新 torch 的 API 不兼容。')
    print('  若要走这条路，建议先用少量数据跑通 scripts/run_finetune.sh smoke 再全量。')
    return 1


if __name__ == '__main__':
    sys.exit(main())
