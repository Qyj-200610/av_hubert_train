#!/usr/bin/env bash
# =============================================================================
# pipeline_lrs2.sh -- AV-HuBERT 在 LRS2 上复现的完整数据流水线
#
# 覆盖：数据列表 -> 抽音频 -> 唇部 ROI -> 数帧 -> 生成 manifest/词表
# 前提：已在 Linux + conda 环境（见 environment.yml）中，且已激活该环境。
#
# 用法：
#   export LRS2_ROOT=/path/to/lrs2/main          # 原始数据集（含 train/val/test）
#   export DATALIST=/path/to/lrs2_datalist       # 招新题目给的列表目录
#   export WORK=/path/to/lrs2_data               # 工作目录（输出都放这里）
#   export FFMPEG=$(which ffmpeg)
#   export DLIB=/path/to/dlib_models             # dlib 模型目录
#   export NSHARD=8                              # 并行分片数
#   bash scripts/pipeline_lrs2.sh all
#
# 也可单独执行某一步：
#   bash scripts/pipeline_lrs2.sh prepare|audio|landmark|mouth|frames|manifest
# =============================================================================

set -euo pipefail

# ---------- 0. 路径与参数 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP="${REPO_ROOT}/avhubert/preparation"

: "${LRS2_ROOT:?请先 export LRS2_ROOT=<LRS2 main 目录>}"
: "${DATALIST:?请先 export DATALIST=<lrs2_datalist 目录>}"
: "${WORK:?请先 export WORK=<工作目录>}"
: "${FFMPEG:?请先 export FFMPEG=<ffmpeg 可执行文件路径>}"
: "${NSHARD:=8}"

# 需要两个 dlib 模型（bzip2 解压后使用）+ 一个参考平均脸：
#   ${DLIB}/mmod_human_face_detector.dat          <- mmod_human_face_detector.dat.bz2
#   ${DLIB}/shape_predictor_68_face_landmarks.dat <- shape_predictor_68_face_landmarks.dat.bz2
#   ${DLIB}/20words_mean_face.npy                 <- 见脚本 scripts/fetch_dlib_models.sh
: "${DLIB:=${WORK}/dlib}"
CNN_DETECTOR="${DLIB}/mmod_human_face_detector.dat"
FACE_PREDICTOR="${DLIB}/shape_predictor_68_face_landmarks.dat"
MEAN_FACE="${DLIB}/20words_mean_face.npy"

# =============================================================================
# 前置检查 1：不要把数据放在根分区（恒源云 / AutoDL 等平台的常见坑）
#   恒源云实例的根分区 / 通常只有约 20G 可用，而 /hy-tmp 才是大容量高速盘。
#   若把 WORK 设成 /root/... 会在抽取唇部 ROI 时把磁盘写满，
#   而且失败信息往往很难看出是磁盘问题。
# =============================================================================
check_disk_space() {
  local target="$1"
  local avail_gb
  avail_gb=$(df -BG --output=avail "${target}" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [[ -z "${avail_gb}" ]]; then
    echo "警告：无法读取 ${target} 的可用空间，跳过检查"
    return 0
  fi

  echo "可用空间检查：${target} 剩余 ${avail_gb} GB"

  # 根分区且剩余不足 30G -> 大概率会中途写满
  if [[ "${avail_gb}" -lt 30 ]]; then
    case "${target}" in
      /root*|/|/root)
        cat >&2 <<EOF

=============================================================================
错误：WORK 落在根分区且只剩 ${avail_gb} GB。

这是云平台最常见的坑：实例根分区一般只有约 20G，装不下预处理产物。
请把 WORK 改到大容量盘，例如恒源云的 /hy-tmp：

    export WORK=/hy-tmp/lrs2_data
    export DLIB=/hy-tmp/dlib

并把数据集也放在 /hy-tmp 下，例如：

    export LRS2_ROOT=/hy-tmp/main
    export DATALIST=/hy-tmp/lrs2_datalist

（恒源云：/hy-tmp 是实例本地高速盘，关机 24 小时后会被清空，
  重要结果请及时用 oss cp 传到「个人数据」。）
=============================================================================
EOF
        exit 1
        ;;
    esac
    echo "警告：可用空间仅 ${avail_gb} GB，预处理产物可能装不下，请留意" >&2
  fi
}

# 前置检查 2：大容量盘提示（恒源云特有）
if [[ -d /hy-tmp ]]; then
  hy_gb=$(df -BG --output=avail /hy-tmp 2>/dev/null | tail -1 | tr -dc '0-9')
  echo "检测到恒源云实例：/hy-tmp 剩余 ${hy_gb:-未知} GB"
  case "${WORK}" in
    /hy-tmp/*) : ;;
    *) echo "提示：建议把 WORK 放在 /hy-tmp 下（当前为 ${WORK}）" >&2 ;;
  esac
fi

mkdir -p "${WORK}"
check_disk_space "${WORK}"

log() { echo -e "\n\033[1;36m==== $* ====\033[0m"; }

# ---------- 1. 数据列表 -> file.list / label.list ----------
step_prepare() {
  log "step 1/6  解析划分列表，生成 file.list / label.list / split.list"
  # 布局：本考核从 AI Studio 下载的 main 实测为 flat
  #       （main/<video_id>/<clip_id>.mp4，无 train/val/test 子目录）
  #
  # LIMIT>0 时只取每个划分的前 N 条（用于先用小样本验证全流程，再跑全量）
  local limit_args=()
  if [[ -n "${LIMIT:-}" && "${LIMIT}" -gt 0 ]]; then
    limit_args=(--max-train "${LIMIT}" --max-valid "${LIMIT}" --max-test "${LIMIT}")
    echo "注意：LIMIT=${LIMIT}，只取每个划分前 ${LIMIT} 条做验证"
  fi

  python "${PREP}/lrs2_prepare.py" \
    --lrs2-root "${LRS2_ROOT}" \
    --datalist  "${DATALIST}" \
    --work-dir  "${WORK}" \
    --layout    "${LAYOUT:-flat}" \
    "${limit_args[@]}"
  echo "总条数: $(wc -l < "${WORK}/file.list")"
  echo "fid 样例: $(head -1 "${WORK}/file.list")"
}

# ---------- 2. 抽取音频（wav, 16kHz 单声道）----------
# 注意：原始 LRS2 音频就是 16kHz。这里统一转成 wav 以匹配
#       hubert_dataset 里 scipy.io.wavfile 的读取方式。
step_audio() {
  log "step 2/6  从 mp4 抽取音频到 ${WORK}/audio"
  local rank="${1:-0}"
  python - "$LRS2_ROOT" "$WORK" "$FFMPEG" "$NSHARD" "$rank" <<'PY'
import os, sys, math, subprocess
from tqdm import tqdm

lrs2_root, work, ffmpeg, nshard, rank = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
fids = [ln.strip() for ln in open(os.path.join(work, 'file.list')) if ln.strip()]
n = math.ceil(len(fids) / nshard)
shard = fids[rank*n:(rank+1)*n]
print(f"rank {rank}/{nshard}: {len(shard)} clips")
for fid in tqdm(shard):
    # fid 与实际目录结构一致（flat 布局下即 <video_id>/<clip_id>），可直接拼路径
    src = os.path.join(lrs2_root, fid + '.mp4')
    dst = os.path.join(work, 'audio', fid + '.wav')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(dst):
        continue
    subprocess.call([ffmpeg, '-i', src, '-f', 'wav', '-ar', '16000', '-ac', '1', '-vn', '-y', dst,
                     '-loglevel', 'quiet'])
PY
}

# ---------- 3. 人脸关键点检测 ----------
step_landmark() {
  log "step 3/6  检测人脸关键点 -> ${WORK}/landmark"
  local rank="${1:-0}"
  for m in "${CNN_DETECTOR}" "${FACE_PREDICTOR}"; do
    [[ -f "$m" ]] || { echo "缺少 dlib 模型: $m（可运行 scripts/fetch_dlib_models.sh 下载）"; exit 1; }
  done
  # 注意：--root 必须是 LRS2 根目录，因为脚本内部按 {root}/{fid}.mp4 取原始视频
  python "${PREP}/detect_landmark.py" \
    --root          "${LRS2_ROOT}" \
    --landmark      "${WORK}/landmark" \
    --manifest      "${WORK}/file.list" \
    --cnn_detector  "${CNN_DETECTOR}" \
    --face_predictor "${FACE_PREDICTOR}" \
    --ffmpeg        "${FFMPEG}" \
    --rank          "${rank}" \
    --nshard        "${NSHARD}"
}

# ---------- 4. 对齐并裁出嘴部 ROI ----------
step_mouth() {
  log "step 4/6  裁切嘴部 ROI -> ${WORK}/video"
  local rank="${1:-0}"
  [[ -f "${MEAN_FACE}" ]] || { echo "缺少参考平均脸: ${MEAN_FACE}（可运行 scripts/fetch_dlib_models.sh 下载）"; exit 1; }
  # 注意：--video-direc 必须是 LRS2 根目录，脚本内部按 {video-direc}/{fid}.mp4 取原始视频
  python "${PREP}/align_mouth.py" \
    --video-direc   "${LRS2_ROOT}" \
    --landmark      "${WORK}/landmark" \
    --filename-path "${WORK}/file.list" \
    --save-direc    "${WORK}/video" \
    --mean-face     "${MEAN_FACE}" \
    --ffmpeg        "${FFMPEG}" \
    --rank          "${rank}" \
    --nshard        "${NSHARD}"
}

# ---------- 5. 统计帧数并合并分片 ----------
step_frames() {
  log "step 5/6  统计音频/视频帧数"
  for ((r=0; r<NSHARD; r++)); do
    python "${PREP}/count_frames.py" \
      --root     "${WORK}" \
      --manifest "${WORK}/file.list" \
      --nshard   "${NSHARD}" \
      --rank     "$r"
  done
  : > "${WORK}/nframes.audio"
  : > "${WORK}/nframes.video"
  for ((r=0; r<NSHARD; r++)); do
    cat "${WORK}/nframes.audio.${r}" >> "${WORK}/nframes.audio"
    cat "${WORK}/nframes.video.${r}" >> "${WORK}/nframes.video"
  done
  echo "音频帧数行数: $(wc -l < "${WORK}/nframes.audio")"
  echo "视频帧数行数: $(wc -l < "${WORK}/nframes.video")"
  if compgen -G "${WORK}/missing.list.*" > /dev/null; then
    echo "警告：存在缺失文件清单，请检查："
    cat "${WORK}"/missing.list.*
    exit 1
  fi
}

# ---------- 6. 生成 manifest / 词表 ----------
step_manifest() {
  log "step 6/6  生成 {train,valid,test}.{tsv,wrd} 与 dict.wrd.txt"
  python "${PREP}/lrs2_manifest.py" \
    --work-dir   "${WORK}" \
    --datalist   "${DATALIST}" \
    --vocab-size "${VOCAB_SIZE:-1000}"
}

# ---------- 入口 ----------
case "${1:-all}" in
  prepare)  step_prepare ;;
  audio)    step_audio "${2:-0}" ;;
  landmark) step_landmark "${2:-0}" ;;
  mouth)    step_mouth "${2:-0}" ;;
  frames)   step_frames ;;
  manifest) step_manifest ;;
  all)
    step_prepare
    if [[ "${NSHARD}" -gt 1 ]]; then
      echo "提示：多分片时建议把 audio/landmark/mouth 按 rank 提交到多卡/多进程并行执行。"
      for ((r=0; r<NSHARD; r++)); do step_audio "$r"; done
      for ((r=0; r<NSHARD; r++)); do step_landmark "$r"; done
      for ((r=0; r<NSHARD; r++)); do step_mouth "$r"; done
    else
      step_audio 0; step_landmark 0; step_mouth 0
    fi
    step_frames
    step_manifest
    ;;
  *)
    echo "用法: bash scripts/pipeline_lrs2.sh [all|prepare|audio|landmark|mouth|frames|manifest] [rank]"
    exit 1
    ;;
esac

log "完成。数据目录：${WORK}/data"
