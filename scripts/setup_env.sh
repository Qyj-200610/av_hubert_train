#!/usr/bin/env bash
# =============================================================================
# setup_env.sh -- 安装 AV-HuBERT 的 Python 依赖（在 conda 环境已创建并激活后运行）
#
# 典型流程：
#   conda env create -f environment.yml
#   conda activate avhubert
#   bash scripts/setup_env.sh
#
# 做四件事：
#   1. 安装 av_hubert 自身的 pip 依赖（opencv/sentencepiece/editdistance/...）
#   2. 编译并安装 fairseq（可编辑模式，**关键步骤**，会编译 C++/CUDA 扩展）
#   3. 安装 fairseq 训练/解码用到的额外包（omegaconf、hydra-core 等由 fairseq 带上）
#   4. 自检：import 各关键库并打印版本，最后确认 GPU 可用
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log() { echo -e "\n\033[1;36m==== $* ====\033[0m"; }

log "0. 环境检查"
python -V
python -c "import sys; print('executable:', sys.executable)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "avhubert" ]]; then
  echo "⚠️  当前 conda 环境是 '${CONDA_DEFAULT_ENV:-<无>}'，建议先 conda activate avhubert"
fi

log "1. 安装 av_hubert 的 pip 依赖"
# 注意：不要直接 pip install -r requirements.txt —— 里面的 opencv-python==4.5.4.60
# 在 py3.8 上没有 wheel。这里显式给出可用版本。
pip install \
  python-speech-features==0.6 \
  scipy==1.10.1 \
  opencv-python==4.9.0.80 \
  sentencepiece==0.1.96 \
  editdistance==0.6.0

log "2. 安装数据预处理依赖"
pip install \
  pydub==0.25.1 \
  scikit-video==1.1.11 \
  submitit==1.4.1 \
  jiwer==3.0.3

# ---------------------------------------------------------------------------
# 2b. dlib —— 单独处理，不要和上面混在一条命令里
#
# 为什么单独拿出来（实测踩过的坑）：
#   * PyPI 上 dlib 只有 sdist，**没有任何 cp38-linux 预编译 wheel**，
#     所以 `pip install dlib` 必然触发本地 C++ 编译，而编译常常失败。
#   * 若把 dlib 和其它包写在同一条 pip 命令里，dlib 编译失败会**导致整条
#     命令中止**，opencv / sentencepiece 等全部装不上（这个坑我踩过）。
#   * 部分平台的 conda 只有 classic solver，`conda install -c conda-forge dlib`
#     求解依赖树极慢（实测 >20 分钟不收敛），`--no-deps` 也仍会走求解阶段。
#
# 可靠做法：直接下载 conda-forge 的预编译包并解压（conda 包就是 bz2 归档），
# 再用 apt 补齐它需要的 BLAS 动态库。
# ---------------------------------------------------------------------------
install_dlib_conda() {
  local ENV_PREFIX
  ENV_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
  local PYVER
  PYVER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  local PKG="dlib-19.24.2-py38h21aafda_0.tar.bz2"
  local BASE="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64"
  local FALLBACK="https://conda.anaconda.org/conda-forge/linux-64"

  if python -c "import dlib" 2>/dev/null; then
    echo "  dlib 已可用，跳过"
    return 0
  fi

  echo "  用「下载 conda 包并解压」的方式安装 dlib（绕开求解器）"
  local TMP
  TMP="$(mktemp -d)"
  # 注意 curl 必须带 --fail：否则镜像返回 404/5xx 时 curl 会把错误页写进 $PKG
  # 并**返回 0**，于是下面 `||` 后的备用镜像永远不会被尝试，最后 tar 报"解压失败"，
  # 把"镜像没有这个包"误导成"包坏了"。（fetch_dlib_models.sh 用的是 --fail，口径一致。）
  # 这条注释必须放在 `( ... )` 之外：续行符 \ 后面紧跟注释会把 && 链切断（已踩过）。
  ( cd "$TMP" \
    && { curl -fsSL -o "$PKG" "$BASE/$PKG" || curl -fsSL -o "$PKG" "$FALLBACK/$PKG"; } \
    && tar -xjf "$PKG" \
    && cp -r lib/python${PYVER}/site-packages/* "$ENV_PREFIX/lib/python${PYVER}/site-packages/" ) \
    || { echo "  dlib 包下载/解压失败（两个镜像都试过了；请检查网络或手工下载 $PKG）"; rm -rf "$TMP"; return 1; }
  rm -rf "$TMP"

  # 补齐 BLAS：dlib 的 .so 依赖 libcblas.so.3 / liblapack.so.3
  # 注意 Ubuntu 20.04 上 libcblas3 已被 libatlas3-base 取代，后者正好提供这两个
  export DEBIAN_FRONTEND=noninteractive
  apt-get install -y libatlas3-base >/dev/null 2>&1 || true

  if python -c "import dlib" 2>/dev/null; then
    echo "  dlib 安装成功: $(python -c 'import dlib; print(dlib.__version__)')"
    return 0
  fi

  echo "  dlib 仍不可用。诊断当前缺失的动态库："
  local SO
  SO="$(find "$ENV_PREFIX/lib/python${PYVER}/site-packages" -maxdepth 1 -name '_dlib*.so' | head -1)"
  ldd "$SO" 2>/dev/null | grep -i 'not found' || echo "  (ldd 未见缺失)"
  echo "  备选：pip install dlib==19.24.2（会本地编译，需 gcc + cmake）"
  return 1
}

log "2b. 安装 dlib（绕开 pip 编译与 conda 求解）"
install_dlib_conda || { echo "dlib 未装好；预处理脚本 detect_landmark.py / align_mouth.py 会失败"; }

log "3. 编译安装 fairseq（这一步最慢，通常 5-15 分钟）"
echo "numpy 版本（必须是 1.x）: $(python -c 'import numpy; print(numpy.__version__)')"
pip install --editable ./fairseq
# 若编译报错，常见原因与对策：
#   * numpy 2.x -> pip install "numpy<2"
#   * 缺 nvcc / CUDA toolkit -> 设置 export CUDA_HOME=/usr/local/cuda
#   * 缺编译器 -> conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11
#   * 只想跑 CPU（慢，仅用于验证流程）-> export FORCE_CUDA=0

log "4. 自检"
python - <<'PY'
import importlib, sys
mods = ['torch', 'torchaudio', 'fairseq', 'numpy', 'scipy', 'cv2',
        'sentencepiece', 'editdistance', 'python_speech_features', 'skvideo',
        'dlib']
fail = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print('  OK   %-24s %s' % (m, getattr(mod, '__version__', '')))
    except Exception as e:
        print('  FAIL %-24s %s' % (m, e))
        fail.append(m)

import torch
print()
print('  torch      :', torch.__version__)
print('  cuda 可用  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU 数量   :', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print('    [%d] %s' % (i, torch.cuda.get_device_name(i)))
else:
    print('  ⚠️  CUDA 不可用，无法进行微调（只能做数据预处理）')

if fail:
    print()
    print('  ✗ 以下模块导入失败：', ', '.join(fail))
    sys.exit(1)
print()
print('  全部依赖 OK')
PY

log "完成"
echo "下一步："
echo "  bash scripts/fetch_dlib_models.sh \$DLIB"
echo "  bash scripts/pipeline_lrs2.sh all"
echo "  bash scripts/run_finetune.sh smoke"
