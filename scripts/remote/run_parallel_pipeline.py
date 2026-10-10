#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_parallel_pipeline.py -- 执行 docs/预处理并行化方案.md 的编排脚本

用法（本地 Windows 运行，通过 SSH 操作恒源云实例）：
    python scripts/remote/run_parallel_pipeline.py env        # 查看远端环境变量与机器规格
    python scripts/remote/run_parallel_pipeline.py stop       # 停止正在运行的串行流水线
    python scripts/remote/run_parallel_pipeline.py upload     # 上传改动后的 pipeline_lrs2.sh 并语法检查
    python scripts/remote/run_parallel_pipeline.py validate   # 步骤 1：小样本并行验证（独立 WORK 目录，不动现有产物）
    python scripts/remote/run_parallel_pipeline.py full       # 步骤 3：全量并行运行（保留已完成的音频）
    python scripts/remote/run_parallel_pipeline.py monitor    # 步骤 4：查看进度
    python scripts/remote/run_parallel_pipeline.py verify     # 步骤 5：验收

设计说明
--------
方案原文步骤 1 用 `rm -rf /hy-tmp/lrs2_data` + LIMIT=30 做验证，会丢弃已 100%
完成的音频（48,164 个 wav）。但 step_audio 内部有 `if os.path.isfile(dst): continue`，
产物按 fid 命名、各 rank 切片不重叠，因此改为：
  * 小样本验证落在独立目录 /hy-tmp/lrs2_parcheck（互不影响）；
  * 全量运行继续用 /hy-tmp/lrs2_data，音频阶段秒级跳过，只重跑 landmark/mouth。
    （detect_landmark.py 无跳过逻辑，landmark 必然整段重算，并行后仅需十几分钟。）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # gh_ssh/train_optimize 就在同目录
from gh_ssh import connect, run  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_LOCAL = os.environ.get('AVH_REPO_LOCAL') or _REPO_ROOT
REPO_REMOTE = '/hy-tmp/av_hubert_train'
WORK = '/hy-tmp/lrs2_data'          # 全量工作目录（已有 audio 100%）
WORK_TEST = '/hy-tmp/lrs2_parcheck'  # 小样本验证目录（独立，不污染全量产物）
PY = '/usr/local/miniconda3/envs/avhubert/bin/python'
TOTAL = 48164


def sh(cli, cmd, **kw):
    return run(cli, cmd, **kw)


def cmd_env(cli):
    sh(cli, 'echo "--- avh_env.sh ---"; cat /hy-tmp/avh_env.sh; '
            'echo; echo "--- 机器规格 ---"; nproc; free -g | head -2; '
            'echo "--- 磁盘 ---"; df -hT /hy-tmp | tail -1; '
            'echo "--- 现有产物 ---"; du -sh %s/* 2>/dev/null' % WORK,
       timeout=300, label='远端环境与规格')


def cmd_stop(cli):
    # ★ 必须用 [p] / [d] / [a] 这种方括号写法。
    #   整条命令是通过 `bash -c '<字符串>'` 执行的，**外层 shell 自己的命令行里
    #   就含有 "pipeline_lrs2.sh" 这些字面量**；`pkill -f` / `pgrep -f` 匹配的是
    #   完整命令行、而且只排除 pgrep 自己，于是：
    #     - `pkill -f "pipeline_lrs2.sh"` 会把执行它的那个 bash 一起杀掉 →
    #       后面几条 pkill 和收尾的 echo 都不会执行，输出看起来还像"成功"；
    #     - `pgrep -af "..."` 会把外层 shell 当成"残留流水线进程"报出来。
    #   方括号让模式本身不再匹配自己（本项目其它地方 162 行也用了这个技巧）。
    sh(cli, 'pkill -f "[p]ipeline_lrs2.sh" 2>/dev/null; sleep 1; '
            'pkill -f "[d]etect_landmark.py" 2>/dev/null; '
            'pkill -f "[a]lign_mouth.py" 2>/dev/null; sleep 2; '
            'echo "--- 残留进程 ---"; pgrep -af "[p]ipeline_lrs2|[d]etect_landmark|[a]lign_mouth" || echo "无"; '
            'echo "--- load ---"; uptime',
       timeout=300, label='停止串行流水线')


def cmd_upload(cli):
    sftp = cli.open_sftp()
    lp = os.path.join(REPO_LOCAL, r'scripts\pipeline_lrs2.sh')
    rp = REPO_REMOTE + '/scripts/pipeline_lrs2.sh'
    # 先留一份原始串行版本作为回退副本（方案 §6）
    try:
        sftp.stat(rp + '.serial')
        print('  回退副本已存在，跳过备份')
    except IOError:
        remote_old = sftp.open(rp, 'r')
        data = remote_old.read()
        remote_old.close()
        bak = sftp.open(rp + '.serial', 'w')
        bak.write(data)
        bak.close()
        print('  已备份原始串行版本 -> %s.serial (%d 字节)' % (rp, len(data)))
    sftp.put(lp, rp)
    print('  已上传 %s (%d 字节)' % (rp, os.path.getsize(lp)))
    sftp.close()

    sh(cli, 'cd %s && bash -n scripts/pipeline_lrs2.sh && echo "语法检查 OK"; '
            'echo "--- 关键片段 ---"; grep -n "run_ranks_parallel\\|wait \\"\\${pids\\[" scripts/pipeline_lrs2.sh' % REPO_REMOTE,
       timeout=180, tail=20, label='上传并语法检查')


def cmd_validate(cli):
    cmd = (
        'source /hy-tmp/avh_env.sh; '
        'export WORK=%s; export LIMIT=30; '
        'rm -rf "$WORK"; '
        'cd %s && bash scripts/pipeline_lrs2.sh all; rc=$?; '
        'echo "=== 退出码: $rc ==="; '
        'echo "--- 产物数量 ---"; '
        'for d in audio landmark video; do echo "$d: $(find $WORK/$d -type f 2>/dev/null | wc -l)"; done; '
        'wc -l $WORK/file.list $WORK/split.list $WORK/nframes.audio $WORK/nframes.video 2>/dev/null; '
        'echo "data 文件数: $(ls $WORK/data 2>/dev/null | wc -l)"; ls $WORK/data 2>/dev/null; '
        'echo "--- train.tsv 前两行 ---"; head -2 $WORK/data/train.tsv 2>/dev/null; '
        'awk -F"\\t" \'NR==2{print "第二行列数: " NF}\' $WORK/data/train.tsv 2>/dev/null; '
        'exit $rc'
    ) % (WORK_TEST, REPO_REMOTE)
    rc, out = sh(cli, cmd, timeout=3600, tail=40, label='步骤1：小样本并行验证')
    return rc, out


def cmd_full(cli):
    cmd = (
        'source /hy-tmp/avh_env.sh; '
        'export WORK=%s; unset LIMIT; '
        'cd %s && setsid bash scripts/pipeline_lrs2.sh all '
        '</dev/null >/hy-tmp/pipeline_parallel.log 2>&1 & '
        'sleep 8; echo "--- 已启动，当前进程 ---"; '
        'pgrep -af "[p]ipeline_lrs2" | head; '
        'echo "--- 日志开头 ---"; head -20 /hy-tmp/pipeline_parallel.log'
    ) % (WORK, REPO_REMOTE)
    sh(cli, cmd, timeout=300, tail=30, label='步骤3：全量并行运行（后台）')


def cmd_monitor(cli):
    sh(cli, 'L=/hy-tmp/pipeline_parallel.log; '
            'echo "--- 进程 ---"; pgrep -c -f "[p]ipeline_lrs2" || echo 0; '
            'echo "--- 阶段日志 ---"; grep -a "====" $L | tail -6; '
            'echo "--- 进度(TQDM 最后一行) ---"; tail -c 2000 $L | tr "\\r" "\\n" | grep -aE "it/s|it\\]" | tail -3; '
            'echo "--- 错误 ---"; grep -aiE "错误|error|Traceback" $L | tail -5 || echo "无"; '
            'echo "--- 产物 ---"; '
            'for d in audio landmark video; do echo "$d $(find %s/$d -type f 2>/dev/null | wc -l)"; done; '
            'echo "CPU 负载: $(uptime | sed \'s/.*load average/load average/\')"; '
            'echo "可用空间: $(df -BG --output=avail /hy-tmp | tail -1)"' % WORK,
       timeout=300, tail=30, label='步骤4：进度监控')


def cmd_verify(cli):
    # 注意：这里不能用 % 格式化——内嵌的 Python 代码本身含 %s / %d / % 运算，
    # 会被外层格式化吃掉（实测报 "not enough arguments for format string"）。
    # 改用 __TOKEN__ 占位替换。
    cmd = (
        'W=__WORK__; cd $W; echo "=== 1. 三类产物文件数（全量期望各 48164）==="; '
        'for d in audio landmark video; do echo "$d: $(find $d -type f | wc -l)"; done; '
        'echo "=== 2. 计数/列表行数（全量期望各 48164）==="; '
        'wc -l file.list split.list nframes.audio nframes.video; '
        'echo "=== 3. data 目录（期望 9 项）==="; ls data; echo "data 文件数: $(ls data | wc -l)"; '
        'echo "=== 4. train.tsv 首行与列数 ==="; head -2 data/train.tsv; '
        'awk -F"\\t" \'NR==2{print "第二行列数: " NF}\' data/train.tsv; '
        'echo "=== 5. dict.wrd.txt 行数（全量期望约 1000）==="; wc -l data/dict.wrd.txt; '
        'echo "=== 6. 逐条 fid 覆盖检查（方案未要求，更强）==="; '
        '__PY__ - <<\'PY\'\n'
        'import os\n'
        'W = "__WORK__"\n'
        'fids = [l.strip() for l in open(os.path.join(W, "file.list")) if l.strip()]\n'
        'print("file.list 条数: " + str(len(fids)) + " 去重后: " + str(len(set(fids))))\n'
        'miss = {"audio": [], "landmark": [], "video": []}\n'
        'for f in fids:\n'
        '    for d, ext in (("audio", ".wav"), ("landmark", ".pkl"), ("video", ".mp4")):\n'
        '        if not os.path.isfile(os.path.join(W, d, f + ext)):\n'
        '            miss[d].append(f)\n'
        'for d in miss:\n'
        '    print("缺失 " + d + ": " + str(len(miss[d])) + "  " + str(miss[d][:5]))\n'
        'na = sum(int(l) for l in open(os.path.join(W, "nframes.audio")) if l.strip())\n'
        'nv = sum(int(l) for l in open(os.path.join(W, "nframes.video")) if l.strip())\n'
        'print("总音频帧数: " + str(na) + "  总视频帧数: " + str(nv))\n'
        'PY\n'
        'echo "=== 7. 缺失清单（不应存在）==="; ls missing.list.* 2>/dev/null || echo "无 missing.list.*"; '
        'echo "=== 8. 流水线日志收尾 ==="; tail -6 /hy-tmp/pipeline_parallel.log | tr "\\r" "\\n" | tail -4; '
        'echo "=== 9. 是否仍在跑 ==="; pgrep -c -f "[p]ipeline_lrs2" || echo 0'
    ).replace('__WORK__', WORK).replace('__PY__', PY)
    sh(cli, cmd, timeout=1800, tail=60, label='步骤5：验收')


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else 'env').lower()
    fn = {'env': cmd_env, 'stop': cmd_stop, 'upload': cmd_upload,
          'validate': cmd_validate, 'full': cmd_full,
          'monitor': cmd_monitor, 'verify': cmd_verify}.get(which)
    if fn is None:
        print(__doc__)
        return 2
    cli = connect(verbose=False)
    try:
        # ★ 必须把子命令的返回值透传出去：原来写成 `fn(cli)` 然后无条件 `return 0`，
        #   于是 `validate` 里精心写的 `exit $rc` / 小样本校验失败，在本地看仍然是
        #   退出码 0 —— `... validate && collect_results.py` 这类串联会把失败当成功。
        rc = fn(cli)
    finally:
        cli.close()
    if isinstance(rc, int):
        return rc
    # 有些子命令返回的是 (rc, out) 之类的元组，取第一个元素
    if isinstance(rc, tuple) and rc and isinstance(rc[0], int):
        return rc[0]
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    sys.exit(main())
