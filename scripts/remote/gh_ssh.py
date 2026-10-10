#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gh_ssh.py -- 恒源云实例 SSH 连接辅助（统一处理网关限流）

背景
----
gpushare 的接入网关对**并发/频繁连接**敏感。实测触发后会返回：

    SSHException: Error reading SSH protocol banner

此时 TCP 端口仍然可达（`Test-NetConnection` 成功），只是 SSH 会话层被拒。
这类失败在长传输之后、或短时间内多次连接时尤其容易出现。

因此所有脚本都应通过本模块建连，避免各自实现重试逻辑。
等待一段时间（常见 30–90 秒）后通常即可恢复。
"""

import os
import time

import paramiko

# ---------------------------------------------------------------------------
# 凭据来自环境变量，**不再硬编码**（原先明文口令直接写在源码里，属安全问题）。
# 设置方式（推荐写进 ~/.avh_cred 并 chmod 600，或直接 export）：
#
#     AVH_HOST=i-2.gpushare.com
#     AVH_PORT=54671
#     AVH_USER=root
#     AVH_PW=<你的口令>
#
# 仓库里的 scripts/avh_env.example.sh 有完整模板。
# ---------------------------------------------------------------------------
def _cred(name, default=''):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip()


HOST = _cred('AVH_HOST')
USER = _cred('AVH_USER')
PASSWORD = _cred('AVH_PW')
try:
    PORT = int(_cred('AVH_PORT') or 0)
except ValueError:
    PORT = 0


def require_cred():
    """缺凭据时给出明确指引，而不是带着空口令去连（那样只会得到迷惑的认证错误）。"""
    missing = [n for n, v in (('AVH_HOST', HOST), ('AVH_PORT', PORT),
                              ('AVH_USER', USER), ('AVH_PW', PASSWORD)) if not v]
    if missing:
        raise SystemExit(
            '缺少实例凭据：%s\n'
            '请先设置环境变量（例如写入 ~/.avh_cred，每行一条 KEY=VALUE，权限 600）：\n'
            '    export AVH_HOST=<实例地址>\n'
            '    export AVH_PORT=<SSH 端口>\n'
            '    export AVH_USER=<用户名>\n'
            '    export AVH_PW=<口令>\n'
            '出于安全考虑，本仓库不再保存明文口令。' % ', '.join(missing))


def connect(retries=8, base_delay=15, verbose=True, timeout=30):
    """建立 SSH 连接，遇到 banner/网络错误时指数退避重试。

    :param retries:    最多尝试次数
    :param base_delay: 首次退避秒数（之后按 ×1.6 递增，上限 120 秒）
    :param verbose:    是否打印重试信息
    :return:           paramiko.SSHClient（已连接）
    :raises:           最后一次的异常
    """
    last = None
    require_cred()
    for i in range(retries):
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                        timeout=timeout, banner_timeout=90, auth_timeout=60,
                        allow_agent=False, look_for_keys=False)
            if verbose and i > 0:
                print('  [已连接，第 %d 次尝试]' % (i + 1))
            return cli
        except Exception as e:
            last = e
            if i == retries - 1:
                break
            delay = min(base_delay * (1.6 ** i), 120)
            if verbose:
                print('  [连接被拒] %s —— %.0f 秒后重试 (%d/%d)'
                      % (str(e).splitlines()[0][:60], delay, i + 1, retries))
            time.sleep(delay)
    raise last


def run(cli, cmd, timeout=3600, tail=None, label=None):
    """执行命令，返回 (exit_code, combined_output)。"""
    if label:
        print('\n' + '=' * 72)
        print('>> ' + label)
        print('=' * 72)
    _, so, se = cli.exec_command(cmd, timeout=timeout)
    # 顺序读 stdout / stderr 在这里是安全的（不需要额外线程）：
    # paramiko 的通道有自己的接收缓冲（stderr 走同一个 channel 的扩展数据，
    # 由 transport 线程持续收进无界 BufferedPipe），所以不会出现 subprocess 那种
    # "stderr 写满 OS 管道 → 远端阻塞 → 本地卡在 stdout.read()" 的死锁。
    out = so.read().decode('utf-8', 'replace')
    err = se.read().decode('utf-8', 'replace')
    rc = so.channel.recv_exit_status()
    text = (out + err).strip()
    if tail:
        text = '\n'.join([l.rstrip() for l in text.splitlines() if l.strip()][-tail:])
    if label:
        for l in text.splitlines():
            print('   ' + l)
        print('   [exit %s]' % rc)
    return rc, text


def status(verbose=True):
    """快速探活：返回 True/False，不抛异常。"""
    try:
        cli = connect(retries=2, base_delay=8, verbose=verbose)
        cli.close()
        return True
    except Exception:
        return False
