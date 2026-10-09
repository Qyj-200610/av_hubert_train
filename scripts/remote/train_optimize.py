#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_optimize.py -- 「减少训练时间」这条线的操作脚本

用法：
    python scripts/remote/train_optimize.py sync          # 上传 fast 配置 + 改过的 run_finetune.sh
    python scripts/remote/train_optimize.py sha           # 校验预训练权重 sha256
    python scripts/remote/train_optimize.py cfg           # 打印预训练 checkpoint 里的模型维度（核对 base/large）
    python scripts/remote/train_optimize.py timed [fast|train] [步数]   # 计时跑 N 步，量每步耗时
    python scripts/remote/train_optimize.py train         # 正式启动快速微调（detached）
    python scripts/remote/train_optimize.py watch         # 看训练日志

设计要点：
  * 用小样本数据集 /hy-tmp/lrs2_parcheck/data（90 条）做计时，不碰全量数据。
  * 计时用 fairseq 自己日志里的 wall（累计秒），比外部计时更准（不含 import/加载权重）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # gh_ssh/train_optimize 就在同目录
from gh_ssh import connect, run  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_LOCAL = os.environ.get('AVH_REPO_LOCAL') or _REPO_ROOT
REPO = '/hy-tmp/av_hubert_train'
PY = '/usr/local/miniconda3/envs/avhubert/bin/python'
CKPT = REPO + '/pretrained/base_vox_iter4.pt'
CKPT_SHA = 'd84ae1bc2cfd701d81f7829ee424fa7d1d5824db27ba64f58ed985e9492edace'
DATA_TEST = '/hy-tmp/lrs2_parcheck/data'
BPE_TEST = '/hy-tmp/lrs2_parcheck/spm1000/spm_unigram1000.model'
TRAIN_LOG = '/hy-tmp/train_fast.log'

UPLOAD = [
    (r'avhubert\conf\finetune\base_lrs2_video_only_fast.yaml',
     'avhubert/conf/finetune/base_lrs2_video_only_fast.yaml'),
    (r'avhubert\conf\finetune\base_lrs2_video_only.yaml',
     'avhubert/conf/finetune/base_lrs2_video_only.yaml'),
    (r'avhubert\conf\finetune\smoke_lrs2.yaml',
     'avhubert/conf/finetune/smoke_lrs2.yaml'),
    (r'scripts\run_finetune.sh', 'scripts/run_finetune.sh'),
    (r'scripts\make_full_config.py', 'scripts/make_full_config.py'),
    (r'scripts\build_frame_cache.py', 'scripts/build_frame_cache.py'),
    (r'scripts\finalize_and_train.sh', 'scripts/finalize_and_train.sh'),
    (r'scripts\run_reproduce.sh', 'scripts/run_reproduce.sh'),
    # 注意：pipeline_lrs2.sh / count_frames.py / lrs2_manifest.py 故意**不在这里**。
    # 当前流水线正在运行（bash 边读边执行脚本），覆盖 pipeline_lrs2.sh 有读到
    # 错位的风险；count_frames/lrs2_manifest 属于第 5/6 步，也要等这一轮跑完再同步。
    (r'avhubert\utils.py', 'avhubert/utils.py'),
    (r'avhubert\hubert_dataset.py', 'avhubert/hubert_dataset.py'),
    (r'avhubert\hubert_pretraining.py', 'avhubert/hubert_pretraining.py'),
    (r'avhubert\preparation\align_mouth.py', 'avhubert/preparation/align_mouth.py'),
    (r'avhubert\preparation\detect_landmark.py', 'avhubert/preparation/detect_landmark.py'),
    (r'avhubert\preparation\count_frames.py', 'avhubert/preparation/count_frames.py'),
    (r'avhubert\preparation\lrs2_manifest.py', 'avhubert/preparation/lrs2_manifest.py'),
    # pipeline_lrs2.sh 之前**刻意不在**列表里：bash 边读边执行运行中的脚本，
    # 覆盖它有读到错位的风险。流水线已于 17:36 结束，现在同步是安全的。
    (r'scripts\pipeline_lrs2.sh', 'scripts/pipeline_lrs2.sh'),
    (r'scripts\setup_env.sh', 'scripts/setup_env.sh'),
    (r'avhubert\preparation\lrs2_prepare.py', 'avhubert/preparation/lrs2_prepare.py'),
    (r'scripts\verify_dataset.py', 'scripts/verify_dataset.py'),
]


def cmd_sync(cli):
    sftp = cli.open_sftp()
    # 原子替换：先传到 <目标>.new，再 mv 覆盖。
    # 直接 sftp.put 覆盖不是原子的——如果远端恰好在读这个文件（例如流水线的某个
    # rank 正要启动 count_frames.py），可能读到写了一半的内容。mv 是原子重命名，
    # 远端要么看到旧版本、要么看到完整的新版本。
    # （bash 例外：运行中的 shell 脚本会边读边执行，本列表刻意不含 pipeline_lrs2.sh，
    #   它必须等当前这轮流水线跑完再同步。）
    pending = []
    for rel, rrel in UPLOAD:
        lp = os.path.join(REPO_LOCAL, rel)
        rp = REPO + '/' + rrel
        sftp.put(lp, rp + '.new')
        pending.append(rrel)
        print('  上传 %-56s (%d 字节)' % (rrel, os.path.getsize(lp)))
    sftp.close()
    if pending:
        mvs = '; '.join('mv -f "%s/%s.new" "%s/%s"' % (REPO, r, REPO, r) for r in pending)
        run(cli, mvs + '; echo "已原子替换 %d 个文件"' % len(pending),
            timeout=300, tail=3, label='原子替换')
    run(cli, 'cd %s && bash -n scripts/run_finetune.sh && echo "run_finetune.sh 语法 OK"; '
             'bash -n scripts/finalize_and_train.sh && echo "finalize_and_train.sh 语法 OK"; '
             'bash -n scripts/run_reproduce.sh && echo "run_reproduce.sh 语法 OK"; '
             'bash -n scripts/setup_env.sh && echo "setup_env.sh 语法 OK"; '
             # 用绝对路径的 python：这里没 source avh_env.sh，PATH 里不一定有 conda 的
             # python（原来用裸 `python` 会报 command not found，检查静默失效）
             '%s -m py_compile avhubert/preparation/count_frames.py avhubert/preparation/lrs2_manifest.py '
             'scripts/verify_dataset.py scripts/build_frame_cache.py '
             '&& echo "python 语法 OK"; '
             '%s scripts/make_full_config.py --help >/dev/null && echo "make_full_config 可执行"' % (REPO, PY, PY),
        timeout=300, tail=10, label='上传后语法检查')


def cmd_sha(cli):
    run(cli, 'ls -l %s; echo "--- 校验 ---"; '
             's=$(sha256sum %s | cut -d" " -f1); echo "远端 $s"; '
             '[ "$s" = "%s" ] && echo "sha256 一致 ✅" || echo "sha256 不一致 ❌"' % (CKPT, CKPT, CKPT_SHA),
        timeout=600, tail=6, label='校验预训练权重')


def cmd_cfg(cli):
    # 用 __TOKEN__ 替换而不是 % 格式化：内嵌 Python 里含 % 运算，会被外层吃掉
    code = (
        'import torch\n'
        'ck = torch.load("__CKPT__", map_location="cpu")\n'
        'print("顶层键: " + str(sorted(ck.keys())))\n'
        'cfg = ck.get("cfg")\n'
        'm = cfg["model"] if isinstance(cfg, dict) and "model" in cfg else cfg\n'
        'keys = ["encoder_layers","encoder_embed_dim","encoder_ffn_embed_dim",\n'
        '        "encoder_attention_heads","decoder_layers","decoder_embed_dim",\n'
        '        "decoder_ffn_embed_dim","decoder_attention_heads"]\n'
        'for k in keys:\n'
        '    v = m.get(k) if hasattr(m, "get") else getattr(m, k, None)\n'
        '    print("  " + k.ljust(26) + str(v))\n'
        't = cfg["task"] if isinstance(cfg, dict) and "task" in cfg else None\n'
        'if t is not None:\n'
        '    for k in ["modalities","labels","normalize"]:\n'
        '        v = t.get(k) if hasattr(t, "get") else getattr(t, k, None)\n'
        '        print("  task." + k.ljust(21) + str(v))\n'
    ).replace('__CKPT__', CKPT)
    run(cli, '%s -c %s' % (PY, _q(code)), timeout=900, tail=22,
        label='预训练 checkpoint 的模型维度')


def _q(s):
    """把多行 python 代码安全地塞进 shell 单引号里。"""
    return "'" + s.replace("'", "'\"'\"'") + "'"


# 计时用的三个变体：fast 是默认；olddec 只把解码器换回原尺寸，其余完全一致，
# 这样两次测出来的差就是「解码器缩小」这一项的净收益。
VARIANTS = {
    'fast': '',
    'olddec': ('model.decoder_layers=9 model.decoder_embed_dim=1024 '
               'model.decoder_ffn_embed_dim=4096 model.decoder_attention_heads=8'),
    # 开预解码帧缓存（构建见 scripts/build_frame_cache.py），用来 A/B 对比数据加载瓶颈
    'cached': 'task.video_cache=' + DATA_TEST.rsplit('/data', 1)[0] + '/frame_cache',
}


def cmd_timed(cli, which='fast', steps=20, data=None, max_tokens=4000, workers=4):
    if which not in VARIANTS:
        print('变体只能是 %s' % list(VARIANTS))
        return
    data = data or DATA_TEST
    run_dir = '/hy-tmp/exp_timed_' + which
    # 计时时关掉存盘与按 epoch 验证：小数据集上「每 2 步一个 epoch」，
    # 而每个 checkpoint 有 1.9 GB、写盘要 5-7 秒，会把每步耗时完全淹没
    # （实测：不关的话 20 步 85 秒里大部分是写盘）。
    # checkpoint.no_save=true 是关键：否则每次计时都会往磁盘写约 4 GB 的 checkpoint。
    no_save = ('checkpoint.no_save=true '
               'checkpoint.save_interval=100000 checkpoint.save_interval_updates=0 '
               'checkpoint.no_epoch_checkpoints=true')
    extra = (VARIANTS[which] + ' ' + no_save).strip()
    sh = (
        'cd %(repo)s; source /hy-tmp/avh_env.sh; '
        'export DATA=%(data)s CKPT=%(ckpt)s EXP=%(run)s; '
        'rm -rf %(run)s; mkdir -p %(run)s; '
        'export MAX_UPDATE=%(steps)d WORKERS=%(workers)d MAX_TOKENS=%(mt)d EXTRA="%(extra)s"; '
        'echo "=== 计时：%(which)s / %(steps)d 步 / DATA=%(data)s / max_tokens=%(mt)d / workers=%(workers)d ==="; '
        's=$(date +%%s); '
        'bash scripts/run_finetune.sh fast > %(run)s/timed.log 2>&1; rc=$?; '
        'e=$(date +%%s); echo "退出码=$rc  外部总耗时=$((e-s)) 秒（含 import 与加载 1.2G 权重）"; '
        'echo "--- fairseq 自报 ---"; grep -a "done training in" %(run)s/timed.log; '
        'echo "--- 数据加载统计（其实有几条被跳过）---"; grep -a "max_keep=" %(run)s/timed.log | tail -2; '
        'grep -a "train_inner\\|train_wall" %(run)s/timed.log | tail -2; '
        'echo "--- 是否有报错 ---"; grep -aiE "error|Traceback|assert" %(run)s/timed.log | tail -5 || echo 无'
    ) % {'repo': REPO, 'data': data, 'ckpt': CKPT, 'run': run_dir,
         'steps': steps, 'extra': extra, 'which': which,
         'mt': max_tokens, 'workers': workers}
    run(cli, sh, timeout=3600, tail=26, label='计时：%s %d 步' % (which, steps))


def cmd_train(cli):
    # 注意必须 export DATA：run_finetune.sh 开头有 : "${DATA:?...}"，
    # 而 avh_env.sh 里并不导出 DATA（只导出 WORK/REPO/...）。
    # 少这一句的话脚本会立刻退出，而 setsid 让"失败"看起来像"已启动"。
    sh = (
        'source /hy-tmp/avh_env.sh; cd %s; '
        'export DATA=${WORK:-/hy-tmp/lrs2_data}/data; '
        'export CKPT=%s EXP=/hy-tmp/exp; '
        '[ -f "$CKPT" ] || { echo "缺少预训练权重"; exit 1; }; '
        '[ -f "$DATA/train.tsv" ] || { echo "缺少 $DATA/train.tsv（预处理还没跑完？）"; exit 1; }; '
        'mkdir -p /hy-tmp/exp; '
        'setsid bash scripts/run_finetune.sh fast '
        '</dev/null >%s 2>&1 & '
        'sleep 8; echo "--- 启动后 8 秒检查 ---"; '
        'if grep -qaE "请先 export|Traceback|Error" %s; then echo "启动失败，日志尾部："; tail -5 %s; else '
        '  echo "已启动，进程数 $(pgrep -c -f "[h]ydra_train")"; fi'
    ) % (REPO, CKPT, TRAIN_LOG, TRAIN_LOG, TRAIN_LOG)
    try:
        run(cli, sh, timeout=90, tail=10, label='启动快速微调（后台）')
    except Exception as e:
        print('  （启动命令通道超时属正常，任务已在后台跑）%s' % str(e).splitlines()[0][:50])


def cmd_decode(cli, gen='test', beam=10, ckpt=None):
    """跑通解码链路：最终要交付的 WER 点数就在这一步产出。

    用小样本集 + 刚跑出来的 20 步 checkpoint 验证「解码入口能跑、能写出 wer.<split>」，
    避免等真实训练几小时后再发现解码路径也有阻塞。
    """
    ck = ckpt or '/hy-tmp/exp_timed_fast/fast/checkpoints/checkpoint_best.pt'
    out = '/hy-tmp/exp_decode'
    sh = (
        'source /hy-tmp/avh_env.sh; cd %(repo)s; '
        'export DATA=%(data)s CKPT=%(baseckpt)s EXP=%(out)s; '
        '[ -f "%(ck)s" ] || { echo "找不到待解码 checkpoint: %(ck)s"; exit 1; }; '
        'rm -rf %(out)s; mkdir -p %(out)s; '
        'echo "=== 解码：beam=%(beam)s subset=%(gen)s ==="; '
        's=$(date +%%s); '
        'BEAM=%(beam)s bash scripts/run_finetune.sh test %(ck)s %(gen)s > %(out)s/decode.log 2>&1; rc=$?; '
        'e=$(date +%%s); echo "退出码=$rc 耗时=$((e-s)) 秒"; '
        'echo "--- wer 结果（文件名是 wer.<哈希>）---"; '
        'for f in %(out)s/decode_%(gen)s/wer.*; do echo "== $f"; head -2 "$f"; done; '
        'echo "--- 产物 ---"; ls %(out)s/decode_%(gen)s/ 2>/dev/null | head; '
        'echo "--- 日志尾部（失败时看这里）---"; tail -18 %(out)s/decode.log'
    ) % {'repo': REPO, 'data': DATA_TEST, 'baseckpt': CKPT, 'out': out,
         'ck': ck, 'beam': beam, 'gen': gen}
    run(cli, sh, timeout=3600, tail=40, label='解码链路验证')


def cmd_cache(cli, work='/hy-tmp/lrs2_parcheck', out='/hy-tmp/lrs2_parcheck/frame_cache',
              verify=20, workers=8, limit=0):
    """构建预解码帧缓存，并做逐位一致性校验。"""
    cmd = (
        'source /hy-tmp/avh_env.sh; cd %(repo)s; '
        'echo "=== 构建帧缓存 ==="; '
        'python scripts/build_frame_cache.py --work-dir %(work)s --out %(out)s '
        '--workers %(workers)d --verify %(verify)d %(limit)s'
    ) % {'repo': REPO, 'work': work, 'out': out, 'workers': workers,
         'verify': verify, 'limit': ('--limit %d' % limit) if limit else ''}
    run(cli, cmd, timeout=7200, tail=18, label='构建并校验帧缓存')


def cmd_synth(cli, reps=100):
    """构造一个"足够大"的合成数据集，用来测真实的每步耗时。

    为什么需要：小样本集只有 30 条训练样本 → 2 步就是一个 epoch，而 fairseq 每个
    epoch 都要重建 DataLoader（重启 worker、重读标签、重跑 load_audio_visual），
    于是"每步耗时"被 epoch 切换开销严重放大。真实数据一个 epoch 有约 573 步，
    这部分开销被摊薄。用复制行数的办法把数据集撑到 ~1 个 epoch，测出来的每步耗时
    才有代表性。

    做法：把 parcheck 的 tsv/wrd 行按 N 倍复制（id 加后缀保证唯一），视频/音频路径
    不变 —— 这样帧缓存的键依然命中，跑的就是与真实训练完全相同的代码路径。
    """
    code = (
        'import os\n'
        'src = "/hy-tmp/lrs2_parcheck/data"\n'
        'dst = "/hy-tmp/lrs2_synth/data"\n'
        'os.makedirs(dst, exist_ok=True)\n'
        'N = %d\n'
        'for split, reps in (("train", N), ("valid", 1), ("test", 1)):\n'
        '    lines = open(os.path.join(src, split + ".tsv")).read().split("\\n")\n'
        '    root = lines[0]\n'
        '    body = [l for l in lines[1:] if l.strip()]\n'
        '    wrds = [l.rstrip("\\n") for l in open(os.path.join(src, split + ".wrd")) if l.strip()]\n'
        '    assert len(wrds) == len(body), (split, len(wrds), len(body))\n'
        '    n = 0\n'
        '    with open(os.path.join(dst, split + ".tsv"), "w") as f1, \\\n'
        '         open(os.path.join(dst, split + ".wrd"), "w") as f2:\n'
        '        f1.write(root + "\\n")\n'
        '        for k in range(reps):\n'
        '            for i, l in enumerate(body):\n'
        '                items = l.split("\\t")\n'
        '                items[0] = items[0] + "_r" + str(k)\n'
        '                f1.write("\\t".join(items) + "\\n")\n'
        '                f2.write(wrds[i] + "\\n")\n'
        '                n += 1\n'
        '    print(split, n)\n'
        'os.makedirs("/hy-tmp/lrs2_synth", exist_ok=True)\n'
        'spm = "/hy-tmp/lrs2_synth/spm1000"\n'
        'if not os.path.exists(spm):\n'
        '    os.symlink("/hy-tmp/lrs2_parcheck/spm1000", spm)\n'
        'print("完成")\n'
    ) % reps
    run(cli, 'source /hy-tmp/avh_env.sh; %s -c %s' % (PY, _q(code)),
        timeout=900, tail=8, label='构造合成数据集')


def cmd_watch(cli):
    run(cli, 'L=%s; echo "--- 进程 ---"; pgrep -c -f "[h]ydra_train" || echo 0; '
             'echo "--- GPU ---"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; '
             'echo "--- 尾部 ---"; grep -a "train_inner" $L | tail -3; '
             'tail -3 $L' % TRAIN_LOG,
        timeout=300, tail=14, label='训练监控')


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else 'sync'
    cli = connect(verbose=False)
    try:
        if a == 'sync':
            cmd_sync(cli)
        elif a == 'sha':
            cmd_sha(cli)
        elif a == 'cfg':
            cmd_cfg(cli)
        elif a == 'timed':
            which = sys.argv[2] if len(sys.argv) > 2 else 'fast'
            steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
            data = sys.argv[4] if len(sys.argv) > 4 else None
            mt = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
            wk = int(sys.argv[6]) if len(sys.argv) > 6 else 4
            cmd_timed(cli, which, steps, data, mt, wk)
        elif a == 'synth':
            reps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            cmd_synth(cli, reps)
        elif a == 'sync_timed':
            # 一次连接里先同步再计时，避免频繁建连触发网关限流
            which = sys.argv[2] if len(sys.argv) > 2 else 'fast'
            steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
            cmd_sync(cli)
            cmd_timed(cli, which, steps)
        elif a == 'train':
            cmd_train(cli)
        elif a == 'decode':
            gen = sys.argv[2] if len(sys.argv) > 2 else 'test'
            beam = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            ck = sys.argv[4] if len(sys.argv) > 4 else None
            cmd_decode(cli, gen, beam, ck)
        elif a == 'cache':
            work = sys.argv[2] if len(sys.argv) > 2 else '/hy-tmp/lrs2_parcheck'
            out = sys.argv[3] if len(sys.argv) > 3 else work + '/frame_cache'
            verify = int(sys.argv[4]) if len(sys.argv) > 4 else 20
            cmd_cache(cli, work, out, verify)
        elif a == 'watch':
            cmd_watch(cli)
        else:
            print(__doc__)
            return 2
    finally:
        cli.close()
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    sys.exit(main())
