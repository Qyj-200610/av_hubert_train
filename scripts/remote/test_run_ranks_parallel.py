#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_run_ranks_parallel.py -- 单独验证 run_ranks_parallel 的失败传播语义

方案 §4 风险 4 说「任一 rank 失败可能导致整批白跑」，靠 run_ranks_parallel 检查
每个 pid 的退出码来兜底。这条失败路径在小样本验证里没被触发过，
因此这里把函数从脚本里抽取出来，用桩函数单独测：

  1. 全部成功     -> 不终止，退出码 0
  2. 某个 rank 失败 -> 打印是哪個 rank，exit 1
  3. 中间 rank 失败，其余成功 -> 仍然 exit 1（不会因为先 wait 到成功的就放过去）
"""
import os
import sys

# gh_ssh / train_optimize 就在同目录（本脚本已从工作区根目录移入仓库）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gh_ssh import connect, run  # noqa: E402

REPO = '/hy-tmp/av_hubert_train'

TEST = r'''
set -uo pipefail
sed -n '/^run_ranks_parallel()/,/^}/p' %(repo)s/scripts/pipeline_lrs2.sh > /tmp/fn.sh
echo "--- 抽取到的函数行数: $(wc -l < /tmp/fn.sh) ---"
. /tmp/fn.sh

NSHARD=4
step_ok()    { sleep 0.2; return 0; }
step_allbad(){ sleep 0.1; return 7; }
step_mixed() { if [[ "$1" == "2" ]]; then sleep 0.1; return 3; fi; sleep 0.3; return 0; }

echo "=== 用例1: 全部成功（期望：不终止，退出码 0）==="
# 注意：`cmd | tail -3; echo $?` 里的 $? 是 **tail** 的状态，恒为 0，
# 起不到校验作用。要拿到子 shell 的退出码必须先把输出重定向到文件。
( run_ranks_parallel step_ok ) >/tmp/o1.txt 2>&1
rc1=$?
echo "用例1 退出码=$rc1 （期望 0）"
tail -3 /tmp/o1.txt

echo "=== 用例2: 全部失败（期望：4 条错误 + exit 1）==="
( run_ranks_parallel step_allbad ) >/tmp/o2.txt 2>&1; echo "用例2 退出码=$? （期望 1）"; grep -c "失败" /tmp/o2.txt

echo "=== 用例3: 仅 rank 2 失败（期望：exit 1 且点名 rank 2）==="
( run_ranks_parallel step_mixed ) >/tmp/o3.txt 2>&1; echo "用例3 退出码=$? （期望 1）"
grep -o "rank [0-9]* 失败" /tmp/o3.txt | sort -u

echo "=== 用例4: NSHARD=1 边界（期望：正常，exit 0）==="
NSHARD=1
( run_ranks_parallel step_ok ) >/tmp/o4.txt 2>&1; echo "用例4 退出码=$? （期望 0）"
''' % {'repo': REPO}


def main():
    cli = connect(verbose=False)
    try:
        run(cli, TEST, timeout=300, tail=40, label='单测 run_ranks_parallel')
    finally:
        cli.close()
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    sys.exit(main())
