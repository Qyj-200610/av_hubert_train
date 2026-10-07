#!/usr/bin/env bash
# =============================================================================
# run_finetune.sh -- 在 LRS2 上微调 AV-HuBERT 并在 test 上评测
#
# 三种模式：
#   smoke    小规模冒烟测试（20 步），确认环境/数据/模型加载都正常
#   train    正式微调（config: finetune/base_lrs2_video_only.yaml）
#   test     用已微调的 checkpoint 在 test 集上做纯视觉唇读解码 + 算 WER
#
# 用法：
#   export DATA=/path/to/lrs2_data/data          # lrs2_manifest.py 的输出目录
#   export CKPT=/path/to/base_vox_iter4.pt       # 预训练权重
#   export EXP=/path/to/exp                      # 实验输出目录
#   bash scripts/run_finetune.sh smoke
#   bash scripts/run_finetune.sh train
#   bash scripts/run_finetune.sh test  /path/to/exp/train/checkpoint_best.pt
#
# 环境变量（可选）：
#   GPUS=0,1,2,3        指定使用哪些卡（会设置 CUDA_VISIBLE_DEVICES）
#   BPE=$DATA/spm1000/spm_unigram1000.model
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${DATA:?请先 export DATA=<lrs2_manifest.py 输出的 data 目录>}"
: "${CKPT:?请先 export CKPT=<预训练权重 .pt>}"
: "${EXP:=${REPO_ROOT}/exp}"

BPE="${BPE:-${DATA}/spm1000/spm_unigram1000.model}"
[[ -f "${BPE}" ]] || { echo "找不到 sentencepiece 模型: ${BPE}"; exit 1; }
[[ -f "${CKPT}" ]] || { echo "找不到预训练权重: ${CKPT}"; exit 1; }
[[ -f "${DATA}/train.tsv" ]] || { echo "找不到 ${DATA}/train.tsv，请先跑完数据流水线"; exit 1; }

# 选择 GPU
if [[ -n "${GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPUS}"
fi
NGPU="${NGPU:-$(python -c 'import torch;print(max(1,torch.cuda.device_count()))')}"
echo "使用 GPU 数: ${NGPU}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<未设置>}"

MODE="${1:-smoke}"
cd "${REPO_ROOT}"   # common.user_dir=$(pwd) 依赖这一点

case "${MODE}" in
  smoke)
    echo "==== 冒烟测试：20 步 ===="
    python fairseq/fairseq_cli/hydra_train.py \
      --config-dir avhubert/conf --config-name finetune/smoke_lrs2 \
      common.user_dir="$(pwd)" \
      task.data="${DATA}" \
      task.label_dir="${DATA}" \
      task.tokenizer_bpe_model="${BPE}" \
      model.w2v_path="${CKPT}" \
      hydra.run.dir="${EXP}/smoke"
    echo "冒烟测试通过。checkpoint 在 ${EXP}/smoke"
    ;;

  train)
    echo "==== 正式微调：${NGPU} 卡 ===="
    python fairseq/fairseq_cli/hydra_train.py \
      --config-dir avhubert/conf --config-name finetune/base_lrs2_video_only \
      common.user_dir="$(pwd)" \
      distributed_training.distributed_world_size="${NGPU}" \
      distributed_training.nprocs_per_node="${NGPU}" \
      task.data="${DATA}" \
      task.label_dir="${DATA}" \
      task.tokenizer_bpe_model="${BPE}" \
      model.w2v_path="${CKPT}" \
      hydra.run.dir="${EXP}/train"
    echo "微调完成。checkpoint 在 ${EXP}/train"
    ;;

  test)
    FINETUNED="${2:-}"
    [[ -n "${FINETUNED}" ]] || { echo "用法: bash scripts/run_finetune.sh test <微调后的 checkpoint> [gen_subset]"; exit 1; }
    [[ -f "${FINETUNED}" ]] || { echo "找不到 checkpoint: ${FINETUNED}"; exit 1; }
    GEN="${3:-test}"
    OUT="${EXP}/decode_${GEN}"
    echo "==== 在 ${GEN} 集上做纯视觉唇读解码（override.modalities=['video']）===="
    # 官方解码入口是 avhubert/infer_s2s.py（注意：没有 hydra_generate.py）
    python avhubert/infer_s2s.py \
      --config-dir avhubert/conf --config-name s2s_decode \
      common.user_dir="$(pwd)" \
      task.data="${DATA}" \
      task.label_dir="${DATA}" \
      task.tokenizer_bpe_model="${BPE}" \
      dataset.gen_subset="${GEN}" \
      common_eval.path="${FINETUNED}" \
      common_eval.results_path="${OUT}" \
      override.modalities="['video']" \
      hydra.run.dir="${OUT}"
    echo
    echo "================ 测试结果 ================"
    if [[ -f "${OUT}/wer.${GEN}" ]]; then
      cat "${OUT}/wer.${GEN}"
      echo "（这就是题目要求汇报的『点数』：LRS2 ${GEN} 集、纯视觉模态的 WER）"
    else
      echo "未找到 ${OUT}/wer.${GEN}，请检查上面的日志输出。"
    fi
    echo "详细结果：${OUT}（hypo-${GEN}.json 为逐句假设）"
    ;;

  *)
    echo "用法: bash scripts/run_finetune.sh [smoke|train|test] [checkpoint] [gen_subset]"
    exit 1
    ;;
esac
