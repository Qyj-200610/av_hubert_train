#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collect_results.py -- 把实例上的全部原始结果拉回本地 results/ 目录

用法：
    python scripts/remote/collect_results.py

归集内容：
    results/01_预处理/验收原始输出.txt        （现场重跑一遍验收命令，留原始输出）
    results/04_原始产物/训练与评测_全流程日志.log
    results/04_原始产物/训练_hydra日志.log
    results/04_原始产物/解析后的训练配置.yaml
    results/04_原始产物/wer.test.txt / wer.test_mv.txt   （保留原始哈希文件名在文件头）
    results/04_原始产物/hypo_test.json / hypo_test_mv.json
    results/04_原始产物/训练检查点清单.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # gh_ssh/train_optimize 就在同目录
from gh_ssh import connect, run  # noqa: E402
from train_optimize import PY  # noqa: E402   # conda 环境的 python 绝对路径

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.environ.get('AVH_RESULTS_DIR') or os.path.join(_REPO_ROOT, 'results')
RAW = os.path.join(RESULTS, '04_原始产物')
EXP = '/hy-tmp/exp'
WORK = '/hy-tmp/lrs2_data'


def save_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    print('  写入 %s（%d 字节）' % (path, len(text.encode('utf-8'))))


def main():
    os.makedirs(RAW, exist_ok=True)
    cli = connect(verbose=False)

    # ---- 1. 现场重跑预处理验收，留原始输出 ----
    rc, out = run(cli, (
        'W=%s; cd $W; echo "=== 1. 三类产物文件数（期望各 48164）==="; '
        'for d in audio landmark video; do echo "$d: $(find $d -type f | wc -l)"; done; '
        'echo "=== 2. 列表/计数行数 ==="; wc -l file.list split.list nframes.audio nframes.video; '
        'echo "=== 3. data 目录 ==="; ls data; echo "data 文件数: $(ls data | wc -l)"; '
        'echo "=== 4. train.tsv 首行与列数 ==="; head -2 data/train.tsv; '
        'awk -F"\\t" \'NR==2{print "第二行列数: " NF}\' data/train.tsv; '
        'echo "=== 5. 词表 ==="; wc -l data/dict.wrd.txt; '
        'echo "=== 6. test_mv 条数 ==="; wc -l data/test_mv.tsv; '
        'echo "=== 7. 缺失清单 ==="; ls missing.list.* 2>/dev/null || echo "无 missing.list.*"; '
        'echo "=== 8. 产物体积 ==="; du -sh audio landmark video data 2>/dev/null; '
        'echo "=== 9. 总帧数 ==="; awk "{s+=\\$1} END{print \\"总视频帧数: \\" s}" nframes.video'
    ) % WORK, timeout=600)
    save_text(os.path.join(RESULTS, '01_预处理', '验收原始输出.txt'), out + '\n')

    # ---- 2. 文本类原始产物 ----
    for remote, local in (
        ('/hy-tmp/final_run.log', '训练与评测_全流程日志.log'),
        ('%s/fast/hydra_train.log' % EXP, '训练_hydra日志.log'),
    ):
        rc, out = run(cli, 'cat %s 2>/dev/null | tail -c 3000000' % remote, timeout=600)
        if out:
            save_text(os.path.join(RAW, local), out + '\n')
        else:
            print('  [跳过] %s 不存在' % remote)

    # ---- 3. 解析后的训练配置 ----
    # 必须用绝对路径的 python：本脚本不 source avh_env.sh，PATH 里不一定有 conda 的
    # python（踩过：裸 `python` 报 command not found，而这里原来没有 else 分支，
    # 于是**静默少一个交付文件**且毫无提示）
    rc, out = run(cli, 'cd /hy-tmp/av_hubert_train && %s scripts/make_full_config.py '
                       '--src avhubert/conf/finetune/base_lrs2_video_only_fast.yaml '
                       '--out /tmp/_resolved.yaml && cat /tmp/_resolved.yaml' % PY,
                  timeout=600)
    if out and 'decoder_layers' in out:
        save_text(os.path.join(RAW, '解析后的训练配置.yaml'), out + '\n')
    else:
        print('  [失败] 没取到解析后的训练配置（rc=%s）：%s'
              % (rc, (out[-200:].replace('\n', ' ') if out else '（输出为空）')))

    # ---- 4. WER 与逐句假设 ----
    sftp = cli.open_sftp()
    for gen, tag in (('test', 'test'), ('test_mv', 'test_mv')):
        d = '%s/decode_%s' % (EXP, gen)
        rc, listing = run(cli, 'ls %s 2>/dev/null' % d, timeout=300)
        names = [x.strip() for x in listing.splitlines() if x.strip()]
        wer_name = next((n for n in names if n.startswith('wer.')), None)
        hypo_name = next((n for n in names if n.startswith('hypo-')), None)
        if wer_name:
            content = run(cli, 'cat %s/%s' % (d, wer_name), timeout=300)[1]
            save_text(os.path.join(RAW, 'wer.%s.txt' % tag),
                      '# 原始文件名: %s\n%s\n' % (wer_name, content))
        else:
            print('  [缺失] %s 下没有 wer.*' % d)
        if hypo_name:
            lp = os.path.join(RAW, 'hypo_%s.json' % tag)
            sftp.get('%s/%s' % (d, hypo_name), lp)
            print('  拉取 %s/%s -> %s（%d 字节）' % (d, hypo_name, lp, os.path.getsize(lp)))
        else:
            print('  [缺失] %s 下没有 hypo-*.json' % d)
    sftp.close()

    # ---- 5. checkpoint 清单 ----
    rc, out = run(cli, 'ls -l %s/fast/checkpoints/ 2>/dev/null' % EXP, timeout=300)
    save_text(os.path.join(RAW, '训练检查点清单.txt'), out + '\n')

    cli.close()
    print('\n归集完成。目录：%s' % RAW)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    sys.exit(main())
