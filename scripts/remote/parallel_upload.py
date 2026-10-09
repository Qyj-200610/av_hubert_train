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
parallel_upload.py -- 分片并行上传大文件到恒源云实例（带退避与连接数自适应）

经验教训（本脚本据此设计）：
  * 单连接 SFTP 实测约 1.9 MB/s；开 8 条连接会被网关直接拒绝
    （Error reading SSH protocol banner），所以连接数要小、且要错开建立。
  * 分片后合并需要 cat，务必保证分片完整，故上传后逐一校验大小。

用法：
    python scripts/remote/parallel_upload.py                       # 默认 3 连接
    python scripts/remote/parallel_upload.py --conns 2
    python scripts/remote/parallel_upload.py --archive <路径> --dest /hy-tmp
    python scripts/remote/parallel_upload.py --resume              # 跳过远端已完整的分片
    python scripts/remote/parallel_upload.py --merge-only          # 只合并校验，不上传

上传完成后脚本会自动在远端合并并校验总大小。
"""

import argparse
import math
import os
import sys
import threading
import time

import paramiko

HOST = os.environ.get('AVH_HOST', '').strip()
USER = os.environ.get('AVH_USER', '').strip()
PASSWORD = os.environ.get('AVH_PW', '').strip()
try:
    PORT = int((os.environ.get('AVH_PORT') or '0').strip())
except ValueError:
    PORT = 0
DEFAULT_ARCHIVE = os.environ.get('AVH_ARCHIVE') or r'D:\人工智能组招新题目\lrs2_main.tar.gz'


def require_cred():
    missing = [n for n, v in (('AVH_HOST', HOST), ('AVH_PORT', PORT),
                              ('AVH_USER', USER), ('AVH_PW', PASSWORD)) if not v]
    if missing:
        sys.exit('缺少实例凭据：%s\n请设置 AVH_HOST / AVH_PORT / AVH_USER / AVH_PW 环境变量'
                 '（模板见 av_hubert_train/scripts/avh_env.example.sh）。' % ', '.join(missing))

lock = threading.Lock()
done_bytes = [0]
t_start = [0.0]
errors = []
remote_part_sizes = {}


def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return '%.1f %s' % (n, u)
        n /= 1024.0
    return '%.1f TB' % n


def connect(retries=5, base_delay=8):
    """建立连接，失败则指数退避重试（应对网关限流）。"""
    require_cred()
    last = None
    for i in range(retries):
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                        timeout=30, banner_timeout=60, auth_timeout=60,
                        allow_agent=False, look_for_keys=False)
            return cli
        except Exception as e:
            last = e
            delay = base_delay * (2 ** i)
            print('   [连接失败] %s: %s —— %d 秒后重试 (%d/%d)'
                  % (type(e).__name__, str(e)[:60], delay, i + 1, retries))
            time.sleep(delay)
    raise last


def part_range(idx, nparts, total):
    per = math.ceil(total / nparts)
    start = idx * per
    end = min(start + per, total)
    return start, end


def upload_part(idx, nparts, local, dest, total, stagger):
    time.sleep(stagger * idx)          # 错开建立，避免同时握手被拒
    remote = '%s/%s.part%02d' % (dest.rstrip('/'), os.path.basename(local), idx)
    start, end = part_range(idx, nparts, total)
    if start >= end:
        return
    want = end - start

    try:
        cli = connect()
        sftp = cli.open_sftp()

        # 断点续传：远端已有内容就从断点接着传（不是整片重传）
        # 远端 .part 是按顺序写入的，所以文件大小 == 已成功写入的字节数
        already = 0
        try:
            st = sftp.stat(remote)
            already = st.st_size
        except IOError:
            already = 0

        if already == want:
            with lock:
                done_bytes[0] += want
                remote_part_sizes[idx] = want
            print('   part%02d 已完整（%s），跳过' % (idx, human(want)))
            sftp.close()
            cli.close()
            return
        if already > want:
            # 远端比预期还大，说明上次写坏了，从头重来
            print('   part%02d 远端过大（%s > %s），重传' % (idx, human(already), human(want)))
            already = 0

        if already:
            print('   part%02d 从 %.1f%% 处续传' % (idx, 100.0 * already / want))

        CHUNK = 4 * 1024 * 1024
        mode = 'ab' if already else 'wb'
        with open(local, 'rb') as f:
            f.seek(start + already)
            with sftp.open(remote, mode) as rf:
                rf.set_pipelined(True)
                remaining = want - already
                while remaining > 0:
                    n = min(CHUNK, remaining)
                    data = f.read(n)
                    if not data:
                        break
                    rf.write(data)
                    remaining -= len(data)
                    with lock:
                        done_bytes[0] += len(data)

        st = sftp.stat(remote)
        with lock:
            remote_part_sizes[idx] = st.st_size
        if st.st_size != want:
            with lock:
                errors.append('part%02d 大小不符：远端 %d != 期望 %d' % (idx, st.st_size, want))
        sftp.close()
        cli.close()
    except Exception as e:
        with lock:
            errors.append('part%02d: %s: %s' % (idx, type(e).__name__, str(e)[:80]))


def main():
    ap = argparse.ArgumentParser(description='分片并行上传（带退避）')
    ap.add_argument('--archive', default=DEFAULT_ARCHIVE)
    ap.add_argument('--dest', default='/hy-tmp')
    ap.add_argument('--conns', type=int, default=3, help='并发连接数（建议 2-4）')
    ap.add_argument('--stagger', type=float, default=6.0, help='连接建立间隔（秒）')
    ap.add_argument('--merge-only', action='store_true', help='只合并并校验，不上传')
    args = ap.parse_args()

    total = os.path.getsize(args.archive) if not args.merge_only else None
    base = os.path.basename(args.archive)

    if not args.merge_only:
        print('文件  : %s' % args.archive)
        print('大小  : %s' % human(total))
        print('连接数: %d（错开 %.0f 秒）' % (args.conns, args.stagger))
        print('目标  : %s' % args.dest)
        print()

        t_start[0] = time.time()
        threads = [threading.Thread(target=upload_part,
                                    args=(i, args.conns, args.archive, args.dest, total, args.stagger))
                   for i in range(args.conns)]
        for t in threads:
            t.start()

        while any(t.is_alive() for t in threads):
            time.sleep(8)
            el = max(time.time() - t_start[0], 1e-6)
            sp = done_bytes[0] / el
            pct = 100.0 * done_bytes[0] / total
            eta = (total - done_bytes[0]) / max(sp, 1e-6)
            sys.stdout.write('\r  进度 %5.1f%%  %s / %s  %.2f MB/s  剩余 %.0f 分   '
                             % (pct, human(done_bytes[0]), human(total), sp / 1024 ** 2, eta / 60))
            sys.stdout.flush()
        for t in threads:
            t.join()
        el = time.time() - t_start[0]
        print('\r  上传结束，用时 %.1f 分钟（均速 %.2f MB/s）%s'
              % (el / 60, done_bytes[0] / el / 1024 ** 2, ' ' * 30))

        if errors:
            print('\n分片错误：')
            for e in errors:
                print('  ' + e)
            print('\n修正后重跑会跳过已完整的分片（脚本自带断点续传）。')
            sys.exit(1)

    print('\n远端合并分片并校验...')
    cli = connect()
    cmd = ('cd {d} && cat {b}.part* > {b} && rm -f {b}.part* && stat -c %s {b} && ls -lh {b}'
           ).format(d=args.dest, b=base)
    _, so, se = cli.exec_command(cmd, timeout=1800)
    out = so.read().decode('utf-8', 'replace').strip()
    err = se.read().decode('utf-8', 'replace').strip()
    print(out if out else '(无输出)')
    if err:
        print('stderr: ' + err)
    cli.close()

    first = out.split('\n')[0].strip() if out else ''
    if total and first.isdigit() and int(first) == total:
        print('\n✓ 校验通过：远端 %s，与本地完全一致' % human(int(first)))
    elif total:
        print('\n⚠ 大小不一致！本地 %d，远端 %s' % (total, first))


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()
