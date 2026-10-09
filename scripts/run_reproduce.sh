#!/usr/bin/env bash
# =============================================================================
# run_reproduce.sh -- 一键复跑（任务要求 §3.4「预处理、训练、测试的命令行，
#                     最好能一键复跑」）
#
# 串联：数据流水线 -> 预解码帧缓存 -> 微调 -> 在 test / test_mv 上解码出 WER
#
# 用法：
#   # 先准备环境（只做一次）
#   source /hy-tmp/avh_env.sh
#   bash scripts/fetch_dlib_models.sh "$DLIB"
#   mkdir -p pretrained && wget -O pretrained/base_vox_iter4.pt \
#     https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter4.pt
#
#   # A) 小样本冒烟：90 条数据 + 20 步训练 + 解码，约 5 分钟，验证整条链路
#   bash scripts/run_reproduce.sh smoke
#
#   # B) 全量复跑（本次交付用的就是这条）
#   bash scripts/run_reproduce.sh full
#
# 环境变量（可选）：
#   MODALITY=video|av     默认 video（纯视觉，跳过抽音频，省约 1 小时）
#   MAX_UPDATE=8000       微调步数
#   BEAM=10               解码 beam（官方 50；调小可省大量解码时间）
#   NSHARD=8              并行分片数（看 cgroup 配额，不是 nproc）
#   WORK/EXP/LRS2_ROOT/DATALIST/DLIB/FFMPEG  路径覆盖
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 先快照调用方的覆盖值，再 source 环境文件（avh_env.sh 会 export 同名变量，
# 顺序反了的话调用方传的值会被静默丢弃——这个坑本项目踩过一次）
_OVR_WORK="${WORK:-}"; _OVR_EXP="${EXP:-}"; _OVR_MODALITY="${MODALITY:-}"
_OVR_MAX_UPDATE="${MAX_UPDATE:-}"; _OVR_BEAM="${BEAM:-}"; _OVR_NSHARD="${NSHARD:-}"
_OVR_LRS2_ROOT="${LRS2_ROOT:-}"; _OVR_DATALIST="${DATALIST:-}"; _OVR_DLIB="${DLIB:-}"

if [[ -f /hy-tmp/avh_env.sh ]]; then
  # 同 finalize_and_train.sh：avh_env.sh 里用了 ${PYTHONPATH}（未定义时会因 set -u 报错），
  # source 期间临时关掉 -u
  set +u
  # shellcheck disable=SC1091
  source /hy-tmp/avh_env.sh
  set -u
fi

MODE="${1:-full}"
WORK="${_OVR_WORK:-${WORK:-/hy-tmp/lrs2_data}}"
EXP="${_OVR_EXP:-${EXP:-/hy-tmp/exp}}"
MODALITY="${_OVR_MODALITY:-${MODALITY:-video}}"
MAX_UPDATE="${_OVR_MAX_UPDATE:-${MAX_UPDATE:-8000}}"
BEAM="${_OVR_BEAM:-${BEAM:-10}}"
NSHARD="${_OVR_NSHARD:-${NSHARD:-8}}"
LRS2_ROOT="${_OVR_LRS2_ROOT:-${LRS2_ROOT:-}}"
DATALIST="${_OVR_DATALIST:-${DATALIST:-}}"
DLIB="${_OVR_DLIB:-${DLIB:-${WORK}/dlib}}"
# ffmpeg：优先用环境变量，其次从 PATH 找（写死 /usr/bin/ffmpeg 在别的机器上会
# 拿到不存在的路径，而 environment.yml 装的是 conda 的 ffmpeg=4.4）
FFMPEG="${FFMPEG:-$(command -v ffmpeg 2>/dev/null || echo /usr/bin/ffmpeg)}"
CKPT="${CKPT:-${REPO_ROOT}/pretrained/base_vox_iter4.pt}"

log() { echo -e "\n\033[1;36m==== $* ====\033[0m"; }

cd "${REPO_ROOT}" || exit 1

[[ -n "${LRS2_ROOT}" && -n "${DATALIST}" ]] || {
  echo "请先 export LRS2_ROOT=<LRS2 main 目录> 与 DATALIST=<lrs2_datalist 目录>" >&2; exit 1; }
[[ -f "${CKPT}" ]] || { echo "找不到预训练权重：${CKPT}" >&2; exit 1; }
[[ -n "${FFMPEG}" && -x "${FFMPEG}" ]] || {
  echo "找不到可执行的 ffmpeg（当前 FFMPEG='${FFMPEG}'）。请 export FFMPEG=<ffmpeg 绝对路径>。" >&2; exit 1; }

# 关键：把 REPO / CKPT 显式传给子脚本。finalize_and_train.sh 里 REPO 的默认值是
# "/hy-tmp/av_hubert_train"，本仓库不在那个位置时它会直接 cd 失败退出。
# （子脚本用的是 ${_OVR_X:-${X:-默认}} 的快照写法，这里 export 就能生效。）
export REPO="${REPO_ROOT}" CKPT

export LRS2_ROOT DATALIST WORK FFMPEG DLIB NSHARD MODALITY

if [[ "${MODE}" == "smoke" ]]; then
  # 小样本：独立目录，绝不碰全量产物（加后缀前先判断，避免重复调用变成 _smoke_smoke）
  case "${WORK}" in *_smoke) ;; *) export WORK="${WORK}_smoke" ;; esac
  case "${EXP}" in *_smoke) ;; *) export EXP="${EXP}_smoke" ;; esac
  export LIMIT="${LIMIT:-30}"
  export MAX_UPDATE="${SMOKE_STEPS:-20}"
  log "冒烟模式：WORK=${WORK} LIMIT=${LIMIT} 训练 ${MAX_UPDATE} 步"
  rm -rf "${WORK}"
  bash scripts/pipeline_lrs2.sh all || { echo "数据流水线失败" >&2; exit 1; }
  NO_CACHE="${NO_CACHE:-1}" MAX_UPDATE="${MAX_UPDATE}" BEAM="${BEAM}" \
    EXP="${EXP}" WORK="${WORK}" REPO="${REPO_ROOT}" CKPT="${CKPT}" \
    bash scripts/finalize_and_train.sh
  exit $?
fi

log "全量复跑：WORK=${WORK} MODALITY=${MODALITY} MAX_UPDATE=${MAX_UPDATE} BEAM=${BEAM}"
echo "步骤：1) 数据流水线  2) 帧缓存  3) 微调  4) 解码 test / test_mv"
echo "（流水线很慢，建议用 setsid 脱离会话跑；本脚本只负责把四步串起来）"

log "1/4 数据流水线"
bash scripts/pipeline_lrs2.sh all || { echo "数据流水线失败" >&2; exit 1; }

log "2-4/4 帧缓存 -> 微调 -> 解码"
MAX_UPDATE="${MAX_UPDATE}" BEAM="${BEAM}" EXP="${EXP}" WORK="${WORK}" \
  REPO="${REPO_ROOT}" CKPT="${CKPT}" \
  bash scripts/finalize_and_train.sh
