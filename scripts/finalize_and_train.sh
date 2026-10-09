#!/usr/bin/env bash
# =============================================================================
# finalize_and_train.sh -- 预处理跑完之后，一条命令走完「缓存 -> 微调 -> 出 WER」
#
# 为什么要有它：预处理要跑好几小时，训练 1.5 小时，解码几分钟。
# 把这一段串成"一个可脱离会话的后台任务"，就不用依赖任何人的终端还开着，
# 也不会在半夜因为无人盯着而白等。
#
# 用法（建议 setsid 脱离会话）：
#   setsid bash scripts/finalize_and_train.sh </dev/null >/hy-tmp/final_pipeline.log 2>&1 &
#
# 它依次做四件事：
#   1. 等 /hy-tmp/lrs2_data/data/{train,valid,test}.tsv 与 video/ 齐备
#      （并检测"流水线已经死了但产物不齐"这种失败，不会无限等下去）
#   2. 构建预解码帧缓存（scripts/build_frame_cache.py，默认带 30 条逐位校验）
#   3. 用 fast 配置微调（8000 步 / 4000 tokens / 编码器全程冻结），带缓存
#   4. 在 test 与 test_mv 两个口径上解码，打印 WER
#
# 可用环境变量覆盖：
#   MAX_UPDATE=8000   训练步数
#   STOP_HOURS=4      训练墙钟上限（兜底，>0 生效）
#   NO_CACHE=1        不建/不用帧缓存
#   GEN=test_mv       只解码指定口径（默认两个都解）
#   WORK / EXP / REPO 路径覆盖
# =============================================================================
set -uo pipefail

# ---------------------------------------------------------------------------
# 先把调用方传进来的覆盖值存起来，再 source 环境文件。
# 坑（实测踩过）：avh_env.sh 里 export 了 WORK / REPO 等同名变量，
# 如果直接写 WORK="${WORK:-默认}"，source 之后 ${WORK} 已经是环境文件的值，
# 调用方传的 WORK 会被**静默丢弃**——表现为"传了小样本目录却一直在等真实数据"。
# ---------------------------------------------------------------------------
_OVR_REPO="${REPO:-}"
_OVR_WORK="${WORK:-}"
_OVR_EXP="${EXP:-}"
_OVR_CKPT="${CKPT:-}"
_OVR_DATA="${DATA:-}"
_OVR_MAX_UPDATE="${MAX_UPDATE:-}"
_OVR_STOP_HOURS="${STOP_HOURS:-}"
_OVR_NO_CACHE="${NO_CACHE:-}"
_OVR_GEN="${GEN:-}"
_OVR_BEAM="${BEAM:-}"
_OVR_EXTRA="${EXTRA:-}"
_OVR_WAIT="${WAIT_TIMEOUT_H:-}"

# 坑 2（实测踩过，启动即失败）：本脚本开着 `set -u`，而 /hy-tmp/avh_env.sh 第 4 行
# 是 `export PYTHONPATH=...:${PYTHONPATH}` —— 非交互 shell 里 PYTHONPATH 未定义，
# 于是 source 直接报 "PYTHONPATH: unbound variable" 并中断整个脚本。
# 所以 source 期间临时关掉 -u（第三方环境文件不该假设调用方的 shell 选项）。
set +u
source /hy-tmp/avh_env.sh
set -u

REPO="${_OVR_REPO:-${REPO:-/hy-tmp/av_hubert_train}}"
WORK="${_OVR_WORK:-${WORK:-/hy-tmp/lrs2_data}}"
EXP="${_OVR_EXP:-${EXP:-/hy-tmp/exp}}"
CKPT="${_OVR_CKPT:-${CKPT:-${REPO}/pretrained/base_vox_iter4.pt}}"
MAX_UPDATE="${_OVR_MAX_UPDATE:-8000}"
STOP_HOURS="${_OVR_STOP_HOURS:-4}"
NO_CACHE="${_OVR_NO_CACHE:-0}"
GEN="${_OVR_GEN:-test test_mv}"
BEAM="${_OVR_BEAM:-10}"
EXTRA_IN="${_OVR_EXTRA:-}"
WAIT_TIMEOUT_H="${_OVR_WAIT:-12}"
_OVR_DATA="${_OVR_DATA:-}"
unset _OVR_REPO _OVR_WORK _OVR_EXP _OVR_CKPT _OVR_MAX_UPDATE _OVR_STOP_HOURS \
      _OVR_NO_CACHE _OVR_GEN _OVR_BEAM _OVR_EXTRA _OVR_WAIT
CACHE="${WORK}/frame_cache"
MAX_UPDATE="${MAX_UPDATE:-8000}"
STOP_HOURS="${STOP_HOURS:-4}"
NO_CACHE="${NO_CACHE:-0}"
WAIT_TIMEOUT_H="${WAIT_TIMEOUT_H:-12}"

log() { echo "[$(date '+%F %T')] $*"; }

cd "${REPO}" || { log "错误：进不去 ${REPO}"; exit 1; }

# ---------------------------------------------------------------------------
# 1. 等预处理完成
# ---------------------------------------------------------------------------
log "==== 1/4 等待预处理完成 ===="
deadline=$(( $(date +%s) + WAIT_TIMEOUT_H * 3600 ))
need=$(wc -l < "${WORK}/file.list" 2>/dev/null || echo 0)
log "file.list 需要 ${need} 条"
while true; do
  n_video=$(find "${WORK}/video" -type f 2>/dev/null | wc -l)
  if [[ "${n_video}" -ge "${need}" && "${need}" -gt 0 \
        && -f "${WORK}/data/train.tsv" && -f "${WORK}/data/valid.tsv" && -f "${WORK}/data/test.tsv" ]]; then
    log "预处理完成：video=${n_video} / ${need}，data/*.tsv 齐备"
    break
  fi
  # 失败检测：流水线进程没了、但产物还没齐 -> 不干等
  if ! pgrep -f "[p]ipeline_lrs2" >/dev/null 2>&1; then
    sleep 20   # 给 step5/6 一点时间收尾
    if ! pgrep -f "[p]ipeline_lrs2" >/dev/null 2>&1 \
       && [[ ! -f "${WORK}/data/train.tsv" ]]; then
      log "错误：预处理进程已退出，但 data/train.tsv 不存在（video=${n_video}/${need}）"
      log "请检查 /hy-tmp/pipeline_parallel.log 尾部的错误信息。中止。"
      exit 2
    fi
  fi
  if [[ $(date +%s) -gt ${deadline} ]]; then
    log "错误：等待超过 ${WAIT_TIMEOUT_H} 小时仍未齐备（video=${n_video}/${need}），中止。"
    exit 3
  fi
  log "  等待中… video=${n_video}/${need} data/train.tsv=$([[ -f ${WORK}/data/train.tsv ]] && echo 有 || echo 无)"
  sleep 120
done

log "产物清点："
for d in audio landmark video; do log "  ${d}: $(find "${WORK}/${d}" -type f | wc -l)"; done
log "  nframes.audio=$(wc -l < "${WORK}/nframes.audio" 2>/dev/null) nframes.video=$(wc -l < "${WORK}/nframes.video" 2>/dev/null)"
log "  data/: $(ls "${WORK}/data" | wc -l) 项"
df -BG --output=avail /hy-tmp | tail -1

# ---------------------------------------------------------------------------
# 2. 构建预解码帧缓存（可关）
# ---------------------------------------------------------------------------
CACHE_ARGS=()
if [[ "${NO_CACHE}" == "1" ]]; then
  log "==== 2/4 跳过帧缓存（NO_CACHE=1）===="
else
  log "==== 2/4 构建预解码帧缓存 ===="
  need_n=$(cat "${WORK}"/data/{train,valid,test}.tsv 2>/dev/null | wc -l)
  have_n=0
  if [[ -f "${CACHE}/index.npz" ]]; then
    have_n=$(python - "${CACHE}/index.npz" <<'PY'
import sys, numpy as np
try:
    with np.load(sys.argv[1], allow_pickle=False) as d:
        print(len(d["keys"]))
except Exception:
    print(0)
PY
)
  fi
  # 覆盖率检查：--limit 建出来的小缓存、或上次没建完的缓存，光看"文件存在"
  # 会被直接复用，训练时大部分样本悄悄回退到在线解码（正确但慢 20 倍）。
  if [[ "${have_n}" -ge "${need_n}" && "${need_n}" -gt 0 ]]; then
    log "  已有 ${CACHE}（${have_n} >= 需要 ${need_n} 条），跳过"
  else
    if [[ -f "${CACHE}/index.npz" ]]; then
      log "  已有缓存只覆盖 ${have_n} 条，少于需要的 ${need_n} 条，重建"
    fi
    if python scripts/build_frame_cache.py --work-dir "${WORK}" --out "${CACHE}" \
            --workers 8 --verify 30; then
      log "  帧缓存构建并校验通过"
    else
      log "  警告：帧缓存构建失败，改为在线解码继续训练（只是慢一些）"
      NO_CACHE=1
    fi
  fi
  [[ "${NO_CACHE}" == "1" ]] || CACHE_ARGS=("task.video_cache=${CACHE}")
fi

# ---------------------------------------------------------------------------
# 3. 微调
# ---------------------------------------------------------------------------
log "==== 3/4 微调（MAX_UPDATE=${MAX_UPDATE}, STOP_HOURS=${STOP_HOURS}）===="
export DATA="${_OVR_DATA:-${WORK}/data}"
export CKPT
export EXP
unset _OVR_DATA

# 关键：训练目录必须先清干净。fairseq 的 checkpoint.restore_file 默认是
# checkpoint_last.pt，只要该文件存在就会**静默续跑**（checkpoint_utils.py:202-208）。
# 若上一次已经跑完 8000 步，重启后第一轮就 should_stop，训练只走约 1 步就退出、
# 然后我们去解码那份**旧** checkpoint 并把它的 WER 当成这次的结果报出去。
# （这条路径在无人值守时尤其危险。）
if [[ -e "${EXP}/fast/checkpoints/checkpoint_last.pt" ]]; then
  log "检测到旧的 ${EXP}/fast/checkpoints/checkpoint_last.pt，先清空训练目录以免静默续跑"
fi
rm -rf "${EXP}/fast"

t0=$(date +%s)
# EXTRA 用**拼接**而不是数组赋值：直接传 EXTRA=... 会把调用方原有的 EXTRA 顶掉，
# 于是调用方指定的覆盖项（如 model.freeze_finetune_updates=4000）被静默丢弃。
FULL_EXTRA="${EXTRA_IN:+${EXTRA_IN} }${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"}"
log "EXTRA=${FULL_EXTRA:-<空>}"
env MAX_UPDATE="${MAX_UPDATE}" STOP_HOURS="${STOP_HOURS}" EXTRA="${FULL_EXTRA}" \
    bash scripts/run_finetune.sh fast
rc=$?
t1=$(date +%s)
log "微调退出码=${rc}，耗时 $(( (t1 - t0) / 60 )) 分钟"
if [[ "${rc}" -ne 0 ]]; then
  log "错误：微调返回非 0（rc=${rc}）。**中止，不去解码**——否则会把一个崩溃/半训练"
  log "      模型的 WER 当成结果报出去。日志见 ${EXP}/fast 运行输出。"
  exit 5
fi

BEST="${EXP}/fast/checkpoints/checkpoint_best.pt"
[[ -f "${BEST}" ]] || BEST="${EXP}/fast/checkpoints/checkpoint_last.pt"
if [[ ! -f "${BEST}" ]]; then
  log "错误：找不到任何 checkpoint（${EXP}/fast/checkpoints/），无法解码。中止。"
  exit 4
fi
log "用 checkpoint：${BEST}（$(du -h "${BEST}" | cut -f1)）"

# ---------------------------------------------------------------------------
# 4. 解码出 WER（两个口径）
# ---------------------------------------------------------------------------
log "==== 4/4 解码 ===="
decode_rc=0
for g in ${GEN}; do
  log "---- 解码 ${g} ----"
  BEAM="${BEAM}" bash scripts/run_finetune.sh test "${BEST}" "${g}"
  rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    log "错误：解码 ${g} 失败（rc=${rc}）"
    decode_rc=1
    continue
  fi
  # 官方解码写的是 wer.<哈希>，不是 wer.<split>，所以要用通配找
  shopt -s nullglob
  wfiles=("${EXP}/decode_${g}"/wer.*)
  shopt -u nullglob
  if [[ ${#wfiles[@]} -eq 0 ]]; then
    log "错误：解码 ${g} 退出码为 0 但没有产出 wer.* 文件"
    decode_rc=1
  else
    for f in "${wfiles[@]}"; do log "  $(basename "$f"): $(head -1 "$f")"; done
  fi
  log "---- ${g} 完成 ----"
done
if [[ "${decode_rc}" -ne 0 ]]; then
  log "解码存在失败，结果不完整。"
  exit 6
fi
log "全部完成。结果目录：${EXP}/decode_*"
