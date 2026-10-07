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
  dlib==19.24.2 \
  scikit-video==1.1.11 \
  submitit==1.4.1 \
  jiwer==3.0.3

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
        'sentencepiece', 'editdistance', 'python_speech_features', 'skvideo']
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
