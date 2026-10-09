#!/usr/bin/env bash
# =============================================================================
# fetch_dlib_models.sh -- 下载数据预处理所需的 dlib 模型与参考平均脸
#
# 需要三个文件（共约 100MB）：
#   1. mmod_human_face_detector.dat          CNN 人脸检测器（HOG 检测失败时的兜底）
#   2. shape_predictor_68_face_landmarks.dat 68 点人脸关键点预测器
#   3. 20words_mean_face.npy                 参考平均脸（做相似变换对齐用）
#
# 用法：
#   bash scripts/fetch_dlib_models.sh /path/to/dlib
#   # 或直接 export DLIB=/path/to/dlib 后不带参数执行
# =============================================================================

set -euo pipefail

DLIB="${1:-${DLIB:-./dlib}}"
mkdir -p "${DLIB}"
cd "${DLIB}"

echo "下载到: $(pwd)"

# ---------- 1 & 2: dlib 官方模型（bz2 压缩，需解压） ----------
# 三个已知的大小下限（字节）。用它们做"完整性"判断：原来只按文件名判断
# "已存在就跳过"，一次被截断的下载会被永久当成完整文件。
MMOD_BYTES_MIN=7000000          # mmod_human_face_detector.dat 约 7.3 MB
SHAPE_BYTES_MIN=95000000        # shape_predictor_68_face_landmarks.dat 约 99.7 MB
MEANFACE_BYTES_MIN=1000         # 20words_mean_face.npy 约 2.5 KB

file_big_enough() {
  local f="$1" min="$2"
  [[ -f "${f}" ]] || return 1
  local sz
  sz=$(wc -c < "${f}" 2>/dev/null || echo 0)
  [[ "${sz}" -ge "${min}" ]]
}

fetch_bz2() {
  local base="$1" min_bytes="$2"
  if file_big_enough "${base}" "${min_bytes}"; then
    echo "  已存在且大小合理，跳过: ${base} ($(du -h "${base}" | cut -f1))"
    return
  fi
  if [[ -f "${base}" ]]; then
    echo "  已存在的 ${base} 偏小（可能是上次被截断的下载），重新下载"
    rm -f "${base}"
  fi
  echo "  下载 ${base}.bz2 ..."
  # 原来的 http:// 是明文传输，容易被中间设备改内容；dlib.net 支持 https
  curl -L --fail --retry 3 -o "${base}.bz2" "https://dlib.net/files/${base}.bz2"
  echo "  解压 ${base}.bz2 ..."
  bzip2 -d "${base}.bz2"
  if ! file_big_enough "${base}" "${min_bytes}"; then
    echo "  错误：${base} 解压后大小异常（$(wc -c < "${base}" 2>/dev/null || echo 0) 字节 < ${min_bytes}）" >&2
    rm -f "${base}"
    return 1
  fi
  echo "  完成: ${base} ($(du -h "${base}" | cut -f1))"
}

fetch_bz2 mmod_human_face_detector.dat "${MMOD_BYTES_MIN}"
fetch_bz2 shape_predictor_68_face_landmarks.dat "${SHAPE_BYTES_MIN}"

# ---------- 3: 参考平均脸 ----------
# 原来指向 .../raw/master/...：master 是会移动的分支，内容可能被上游改动。
# 想固定版本就设置 MEAN_FACE_COMMIT=<commit sha>，届时用 raw.githubusercontent.com
# 的固定 commit 地址下载，保证任何人任何时候拿到的都是同一份文件。
MEAN_FACE_COMMIT="${MEAN_FACE_COMMIT:-}"
if [[ -n "${MEAN_FACE_COMMIT}" ]]; then
  MEAN_FACE_URL="https://raw.githubusercontent.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/${MEAN_FACE_COMMIT}/preprocessing/20words_mean_face.npy"
else
  # 未指定 commit 时退回到分支地址，但明确提示"未固定版本"
  MEAN_FACE_URL="https://raw.githubusercontent.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/master/preprocessing/20words_mean_face.npy"
  echo "  [提示] 未指定 MEAN_FACE_COMMIT，使用 master 分支（内容可能随上游变动）。"
  echo "         要固定版本：export MEAN_FACE_COMMIT=<commit sha> 后再执行本脚本。"
fi
if file_big_enough "20words_mean_face.npy" "${MEANFACE_BYTES_MIN}"; then
  echo "  已存在且大小合理，跳过: 20words_mean_face.npy"
else
  if [[ -f "20words_mean_face.npy" ]]; then
    echo "  已存在的 20words_mean_face.npy 偏小，重新下载"
    rm -f 20words_mean_face.npy
  fi
  echo "  下载 20words_mean_face.npy ..."
  curl -L --fail --retry 3 -o 20words_mean_face.npy "${MEAN_FACE_URL}"
  if ! file_big_enough "20words_mean_face.npy" "${MEANFACE_BYTES_MIN}"; then
    echo "  错误：20words_mean_face.npy 大小异常，下载可能失败" >&2
    rm -f 20words_mean_face.npy
    exit 1
  fi
  echo "  完成: 20words_mean_face.npy ($(du -h 20words_mean_face.npy | cut -f1))"
fi

echo
echo "全部就绪。请在运行 pipeline_lrs2.sh 前设置："
echo "  export DLIB=$(pwd)"
