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
fetch_bz2() {
  local base="$1"
  if [[ -f "${base}" ]]; then
    echo "  已存在，跳过: ${base}"
    return
  fi
  echo "  下载 ${base}.bz2 ..."
  curl -L --fail --retry 3 -o "${base}.bz2" "http://dlib.net/files/${base}.bz2"
  echo "  解压 ${base}.bz2 ..."
  bzip2 -d "${base}.bz2"
  echo "  完成: ${base} ($(du -h "${base}" | cut -f1))"
}

fetch_bz2 mmod_human_face_detector.dat
fetch_bz2 shape_predictor_68_face_landmarks.dat

# ---------- 3: 参考平均脸 ----------
MEAN_FACE_URL="https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy"
if [[ -f "20words_mean_face.npy" ]]; then
  echo "  已存在，跳过: 20words_mean_face.npy"
else
  echo "  下载 20words_mean_face.npy ..."
  curl -L --fail --retry 3 -o 20words_mean_face.npy "${MEAN_FACE_URL}"
  echo "  完成: 20words_mean_face.npy ($(du -h 20words_mean_face.npy | cut -f1))"
fi

echo
echo "全部就绪。请在运行 pipeline_lrs2.sh 前设置："
echo "  export DLIB=$(pwd)"
