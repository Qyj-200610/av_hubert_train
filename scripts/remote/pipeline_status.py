#!/usr/bin/env python
# ---------------------------------------------------------------------------
# ⚠ 已废弃（DEPRECATED）：请不要再运行本脚本
#
# 本项目当前正确的流程是：
#     bash scripts/pipeline_lrs2.sh all      # 预处理（阶段内按 rank 并行）
#     bash scripts/finalize_and_train.sh      # 帧缓存 -> 微调 -> 解码出 WER
#     python scripts/remote/collect_results.py               # 把结果归集到 results/
#     python watch_final_run.py / watch_pipeline.py   # 监控
#
# 本脚本属于早期迭代版本，上传/取文件请用平台自带 oss 工具或 scripts/remote/ 下的脚本（本脚本的分片续传有已知缺陷）
#
# 确需运行请显式设置环境变量 AVH_ALLOW_DEPRECATED=1。
# ---------------------------------------------------------------------------
import os as _os, sys as _sys
if _os.environ.get("AVH_ALLOW_DEPRECATED") != "1":
    _sys.exit("已废弃脚本：%s\n原因见文件头。若确实要运行，请设 AVH_ALLOW_DEPRECATED=1。" % _os.path.basename(__file__))

# -*- coding: utf-8 -*-
"""
pipeline_status.py -- 查看实例上全量预处理流水线的进度（单次查询，轻量）

用法：
    python scripts/remote/pipeline_status.py
"""
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # gh_ssh/train_optimize 就在同目录
from gh_ssh import connect, run  # noqa: E402

TOTAL = 48164


def main():
    cli = connect(verbose=False)
    print('=' * 66)
    print('LRS2 预处理流水线进度')
    print('=' * 66)

    rc, out = run(cli, 'pgrep -c -f "pipeline_lrs2.sh" 2>/dev/null || echo 0', timeout=60)
    alive = out.strip().splitlines()[-1].strip() if out.strip() else '0'
    print('流水线进程: %s' % ('运行中' if alive not in ('0', '') else '已结束'))

    rc, out = run(cli, 'L=/hy-tmp/pipeline_full.log; '
                       'grep -oE "==== step [0-9]/6[^=]*====" $L 2>/dev/null | tail -3; '
                       'echo "---"; grep -oE "[0-9]+\\.[0-9]+it/s" $L 2>/dev/null | tail -1',
        timeout=120)
    for l in out.splitlines():
        print('  ' + l.strip())

    rc, out = run(cli, 'W=/hy-tmp/lrs2_data; '
                       'for d in audio landmark video; do '
                       '  if [ -d $W/$d ]; then n=$(find $W/$d -type f | wc -l); '
                       '    s=$(du -sh $W/$d 2>/dev/null | cut -f1); echo "$d $n $s"; '
                       '  else echo "$d 0 -"; fi; done; '
                       'for f in nframes.audio nframes.video; do '
                       '  if [ -f $W/$f ]; then echo "$f $(wc -l < $W/$f)"; else echo "$f 0"; fi; done; '
                       'if [ -d $W/data ]; then echo "data $(ls $W/data | wc -l)"; else echo "data 0"; fi; '
                       'df -BG --output=avail /hy-tmp | tail -1',
        timeout=300)
    vals = {}
    avail = '?'
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2 and p[0] in ('audio', 'landmark', 'video', 'data',
                                    'nframes.audio', 'nframes.video'):
            vals[p[0]] = p[1]
        elif len(p) == 1 and p[0].rstrip('G').isdigit():
            avail = p[0]

    print()
    print('产物进度：')
    for k, label in (('audio', '音频'), ('landmark', '关键点'), ('video', '唇部ROI')):
        n = int(vals.get(k, 0))
        print('  %-8s %7d 文件  (%.1f%%)' % (label, n, 100.0 * n / TOTAL))
    print('  帧数文件  nframes.audio=%s  nframes.video=%s'
          % (vals.get('nframes.audio', 0), vals.get('nframes.video', 0)))
    print('  data 目录 %s 项' % vals.get('data', 0))
    print('  可用空间  %s' % avail)

    print()
    if vals.get('data', '0') != '0':
        print('>>> 流水线已完成，可进行微调')
    elif vals.get('video', '0') != '0':
        print('>>> 正在做唇部 ROI（第 4/6 步）')
    elif vals.get('landmark', '0') != '0':
        print('>>> 正在做关键点检测（第 3/6 步，最慢的一步）')
    else:
        print('>>> 正在抽取音频（第 2/6 步）')

    cli.close()


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()
