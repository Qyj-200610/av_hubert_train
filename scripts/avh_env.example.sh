#!/usr/bin/env bash
# =============================================================================
# avh_env.example.sh -- 环境变量模板
#
# 用法：
#   cp scripts/avh_env.example.sh ~/avh_env.sh
#   # 按下面注释改成你自己的路径
#   source ~/avh_env.sh
#
# 或者让脚本自己去读（run_reproduce.sh / finalize_and_train.sh 支持 AVH_ENV）：
#   export AVH_ENV=~/avh_env.sh
#
# 为什么要单独一个文件：原先把这些值写死在 /hy-tmp/avh_env.sh 里，
# 而且 finalize_and_train.sh 会**无条件 source 那个绝对路径**——
# 换一台机器/换一个目录就会带着错误的默认值继续跑。
# 现在：文件不存在会跳过（只提示一句），路径也能用 AVH_ENV 覆盖。
#
# ⚠️ 本文件**不要**放任何口令。实例口令请放进 ~/.avh_cred（权限 600），
#    或直接 export；仓库里不留明文。
# =============================================================================

# ---------- 路径（按需修改）----------
# 注意：下面都用 ${X:-默认} 的写法，这样"先在命令行 export 再 source 本文件"
# 不会被本文件的值顶掉（正是 finalize_and_train.sh 头部记过的那个坑）。
# 仓库自身的位置。不设的话脚本会用"脚本所在目录的上一级"，通常不需要改。
# export REPO=/path/to/av_hubert_train

# 数据预处理的工作目录（所有中间产物与 data/ 都放这里；云上建议放 /hy-tmp）
export WORK="${WORK:-/hy-tmp/lrs2_data}"

# 训练产物目录（checkpoint、解码结果）
export EXP="${EXP:-/hy-tmp/exp}"

# LRS2 main 数据集根目录（指向 main 目录本身，扁平结构）
export LRS2_ROOT="${LRS2_ROOT:-/hy-tmp/main}"

# 招新题目给的三个列表（train.txt / val.txt / test.txt）所在目录
export DATALIST="${DATALIST:-/hy-tmp/lrs2_datalist}"

# dlib 模型目录（由 scripts/fetch_dlib_models.sh 填充）
export DLIB="${DLIB:-/hy-tmp/dlib}"

# ffmpeg。留空则脚本用 `command -v ffmpeg` 自动找；
# 注意 environment.yml 装的是 conda 的 ffmpeg，不一定是 /usr/bin/ffmpeg。
# export FFMPEG=/usr/bin/ffmpeg

# 预训练权重（默认 ${REPO}/pretrained/base_vox_iter4.pt）
# export CKPT="${CKPT:-/hy-tmp/av_hubert_train/pretrained/base_vox_iter4.pt}"

# Python 能 import 到 avhubert / fairseq
# 注意两点：
#   1) 未定义 PYTHONPATH 时要用 :- 兜一下，否则 set -u 的脚本 source 会报
#      "PYTHONPATH: unbound variable" 直接中断（这个坑本项目踩过）。
#   2) **不要**写成 "${PYTHONPATH:-}:..."——那会在最前面留下一个**空组件**，
#      而 Python 把空组件当成"当前目录"，等于把 CWD 悄悄加进 sys.path，
#      可能让工作目录里的同名文件抢先导入。
#      正确写法：自己的路径在前，原有值用 ${PYTHONPATH:+:...} 追加（为空时不加冒号）。
export PYTHONPATH="${REPO:-/hy-tmp/av_hubert_train}${PYTHONPATH:+:${PYTHONPATH}}"

# ---------- 流水线 / 训练行为（一般不用改）----------
# 并行分片数：**看 cgroup 配额，别信 $(nproc)**
# （实测容器 nproc=64 但 CPU 配额只有 8 核；设大了只会增加上下文切换）
export NSHARD="${NSHARD:-8}"

# 模态：video（默认，纯视觉，跳过整步抽音频）| av（音视频）
export MODALITY="${MODALITY:-video}"

# 唇部 ROI 后端：默认是经过逐位校验的快路径。
# 要产出与官方预处理**逐位可比**的 ROI：
#   export WARP_BACKEND=skimage WRITE_BACKEND=png
export WARP_BACKEND="${WARP_BACKEND:-cv2}"
export WRITE_BACKEND="${WRITE_BACKEND:-pipe}"

# 微调步数 / 解码 beam
export MAX_UPDATE="${MAX_UPDATE:-8000}"
export BEAM="${BEAM:-10}"

# ---------- 以下按需打开 ----------
# export MAX_TOKENS=4000          # dataset.max_tokens
# export LR=0.002                 # optimization.lr
# export FREEZE_UPDATES=15000     # 编码器解冻前保持冻结的步数（设成 max_update 的一半即"前半冻结、后半解冻"）
# export WORKERS=8                # dataset.num_workers
# export STOP_HOURS=4             # 训练墙钟上限兜底（>0 生效）
# export NO_CACHE=1               # 不建/不用帧缓存
# export GEN="test test_mv"       # 只解码指定口径
# export LAYOUT=flat              # 数据集目录布局：flat | split
# export VOCAB_SIZE=1000          # sentencepiece 词表大小

# ---------- 实例凭据（只给 scripts/remote/ 下的脚本用）----------
# ⚠️ 不要把口令提交进 git。推荐写进 ~/.avh_cred（chmod 600），每行一条 KEY=VALUE：
#     AVH_HOST=<实例地址>
#     AVH_PORT=<SSH 端口>
#     AVH_USER=<用户名>
#     AVH_PW=<口令>
# 也可以直接用环境变量：
# export AVH_HOST=
# export AVH_PORT=
# export AVH_USER=
# export AVH_PW=
