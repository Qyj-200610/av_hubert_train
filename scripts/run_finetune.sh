#!/usr/bin/env bash
# =============================================================================
# run_finetune.sh -- 在 LRS2 上微调 AV-HuBERT 并在 test 上评测
#
# 四种模式：
#   smoke    小规模冒烟测试（20 步），确认环境/数据/模型加载都正常
#   train    正式微调（config: finetune/base_lrs2_video_only.yaml，忠实对齐官方）
#   fast     快速微调（config: finetune/base_lrs2_video_only_fast.yaml，省时间版）
#   test     用已微调的 checkpoint 在 test 集上做纯视觉唇读解码 + 算 WER
#
# 用法：
#   export DATA=/path/to/lrs2_data/data          # lrs2_manifest.py 的输出目录
#   export CKPT=/path/to/base_vox_iter4.pt       # 预训练权重
#   export EXP=/path/to/exp                      # 实验输出目录
#   bash scripts/run_finetune.sh smoke
#   bash scripts/run_finetune.sh train
#   bash scripts/run_finetune.sh fast            # 单卡赶时间用这个
#   bash scripts/run_finetune.sh test  /path/to/exp/train/checkpoint_best.pt
#
# 环境变量（可选）：
#   GPUS=0,1,2,3        指定使用哪些卡（会设置 CUDA_VISIBLE_DEVICES）
#   BPE=$DATA/spm1000/spm_unigram1000.model
#   fast 模式：MAX_UPDATE / MAX_TOKENS / LR / FREEZE_UPDATES / WORKERS / STOP_HOURS
#   test 模式：BEAM=10（解码 beam，官方 50；调小可省大量解码时间）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# fairseq 的 utils.import_user_module 会这样加载插件：
#     module_parent, module_name = os.path.split(user_dir)   # 见 fairseq/utils.py:479
#     sys.path.insert(0, module_parent); importlib.import_module(module_name)
# 也就是说 user_dir 必须指向**一个可导入的包目录**，包名就是它最后一级目录名。
# 它要导入的是 avhubert/（该目录的 __init__.py 才会 register_task /
# register_model，注册出 av_hubert_pretraining 等）。上游 README 里先 `cd avhubert`
# 再传 `common.user_dir=$(pwd)` 就是这个道理。
# 若误传仓库根目录，会去 import 一个不存在的包，注册表里没有 av_hubert_pretraining，
# 训练直接报：
#     AssertionError: Could not infer task type from {'_name': 'av_hubert_pretraining'...}
USER_DIR="${REPO_ROOT}/avhubert"

: "${DATA:?请先 export DATA=<lrs2_manifest.py 输出的 data 目录>}"
: "${CKPT:?请先 export CKPT=<预训练权重 .pt>}"
: "${EXP:=${REPO_ROOT}/exp}"

# 词表（sentencepiece）自动定位。
# 坑：数据流水线把词表写在 **<work_dir>/spm1000/**（lrs2_manifest.py:98
# `vocab_dir = Path(work_dir) / "spm1000"`），而 DATA 指向的是 <work_dir>/data，
# 两者差一级目录。原写法 ${DATA}/spm1000/... 因此必然找不到词表，
# 训练会在一开始就报「找不到 sentencepiece 模型」而退出。
if [[ -z "${BPE:-}" ]]; then
  for _d in "${DATA}/spm1000" "${DATA}/../spm1000" "${DATA}"; do
    for _f in "${_d}"/spm_unigram*.model; do
      if [[ -f "${_f}" ]]; then BPE="${_f}"; break 2; fi
    done
  done
fi
BPE="${BPE:-${DATA}/spm1000/spm_unigram1000.model}"
echo "词表: ${BPE}"
[[ -f "${BPE}" ]] || { echo "找不到 sentencepiece 模型: ${BPE}（可用 BPE=<路径> 指定）"; exit 1; }

# ---------------------------------------------------------------------------
# 词表规模校验（防"静默 100% WER"）
# 小样本验证（LIMIT=30）时 sentencepiece 会报
#   Vocabulary size too high (N). Please set it to a value <= M.
# 并在 lrs2_manifest.py 里自动降级到 M，但产物**仍然叫 spm_unigram1000.***。
# 若这个降级词表被全量训练用上：训练会一路正常跑完、没有任何报错，
# 而 WER ≈ 100%（目标端根本拼不出词）。这里做一次显式检查。
# ---------------------------------------------------------------------------
_vsize_fn="$(dirname "${BPE}")/../vocab_size.txt"
if [[ -f "${_vsize_fn}" ]]; then
  _recorded="$(tr -dc '0-9' < "${_vsize_fn}")"
  if [[ -n "${_recorded}" && "${_recorded}" -lt 100 ]]; then
    cat >&2 <<EOF
错误：$(realpath "${_vsize_fn}") 记录的词表大小只有 ${_recorded}，
      说明这是小样本（LIMIT）验证时 sentencepiece 自动降级的产物，却仍以
      $(basename "${BPE}") 的名字存在。用它做正式训练会"正常跑完但 WER≈100%"。
      请删掉 $(dirname "${BPE}") 后重跑第 6 步（lrs2_manifest.py）以重建词表。
EOF
    exit 1
  fi
fi
if [[ -f "${DATA}/dict.wrd.txt" ]]; then
  _dict_lines=$(wc -l < "${DATA}/dict.wrd.txt")
  echo "词典 dict.wrd.txt 行数: ${_dict_lines}"
  if [[ "${_dict_lines}" -lt 100 ]]; then
    echo "错误：${DATA}/dict.wrd.txt 只有 ${_dict_lines} 行，词表明显被降级过；" >&2
    echo "      继续训练会得到 WER≈100% 且不报错。请重建词表（见上）。" >&2
    exit 1
  fi
fi
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

# -----------------------------------------------------------------------------
# 生成「自包含配置」再训练。
# 背景：AV-HuBERT 的配置只写了关键项，而 fairseq 的 add_defaults 只补 Any 类型
# 分组（task/model/criterion/optimizer/lr_scheduler），common/checkpoint/dataset/
# distributed_training 的默认值本来靠 fairseq 自带的主配置 config.yaml 提供。
# 一旦用 --config-name finetune/xxx 把主配置换掉，这几段就缺字段，会直接报
#   omegaconf.errors.ConfigAttributeError: Missing key reset_logging
# 所以先用 scripts/make_full_config.py 把它们补全成一份完整配置。
# -----------------------------------------------------------------------------
gen_config() {
  local name="$1"
  local src="avhubert/conf/finetune/${name}.yaml"
  local out="avhubert/conf/finetune/_full_${name}.yaml"
  [[ -f "${src}" ]] || { echo "找不到配置文件: ${src}" >&2; exit 1; }
  python scripts/make_full_config.py --src "${src}" --out "${out}" >&2 || exit 1
  echo "finetune/_full_${name}"
}

case "${MODE}" in
  smoke)
    echo "==== 冒烟测试：20 步 ===="
    CFG_NAME="$(gen_config smoke_lrs2)"
    python fairseq/fairseq_cli/hydra_train.py \
      --config-dir avhubert/conf --config-name "${CFG_NAME}" \
      common.user_dir="${USER_DIR}" \
      task.data="${DATA}" \
      task.label_dir="${DATA}" \
      task.tokenizer_bpe_model="${BPE}" \
      model.w2v_path="${CKPT}" \
      hydra.run.dir="${EXP}/smoke"
    echo "冒烟测试通过。checkpoint 在 ${EXP}/smoke"
    ;;

  train)
    echo "==== 正式微调：${NGPU} 卡 ===="
    CFG_NAME="$(gen_config base_lrs2_video_only)"
    python fairseq/fairseq_cli/hydra_train.py \
      --config-dir avhubert/conf --config-name "${CFG_NAME}" \
      common.user_dir="${USER_DIR}" \
      distributed_training.distributed_world_size="${NGPU}" \
      distributed_training.nprocs_per_node="${NGPU}" \
      task.data="${DATA}" \
      task.label_dir="${DATA}" \
      task.tokenizer_bpe_model="${BPE}" \
      model.w2v_path="${CKPT}" \
      hydra.run.dir="${EXP}/train"
    echo "微调完成。checkpoint 在 ${EXP}/train"
    ;;

  fast)
    # =========================================================================
    # 快速微调：省时间的配置（见 avhubert/conf/finetune/base_lrs2_video_only_fast.yaml）
    # 用法：bash scripts/run_finetune.sh fast
    # 下面这些环境变量都可以不改文件就调（都是可选）：
    #   MAX_UPDATE=8000      优化步数（默认 8000；原 30000）
    #   MAX_TOKENS=4000      批大小（默认 4000；原 1000）
    #   LR=0.002             学习率
    #   FREEZE_UPDATES=8000  冻结编码器的步数；设成 MAX_UPDATE 的一半即"后半程解冻"
    #   WORKERS=8            DataLoader 进程数
    #   STOP_HOURS=2.5       >0 时按累计训练小时硬停（无人值守兜底）
    #   EXTRA="a=b c=d"      任意其它 hydra 覆盖项（空格分隔），例如
    #                        EXTRA="model.decoder_layers=9 model.freeze_finetune_updates=4000"
    # =========================================================================
    echo "==== 快速微调：${NGPU} 卡 ===="
    overrides=()
    [[ -n "${MAX_UPDATE:-}"     ]] && overrides+=("optimization.max_update=${MAX_UPDATE}")
    [[ -n "${MAX_TOKENS:-}"     ]] && overrides+=("dataset.max_tokens=${MAX_TOKENS}")
    [[ -n "${LR:-}"             ]] && overrides+=("optimization.lr=[${LR}]")
    [[ -n "${FREEZE_UPDATES:-}" ]] && overrides+=("model.freeze_finetune_updates=${FREEZE_UPDATES}")
    [[ -n "${WORKERS:-}"        ]] && overrides+=("dataset.num_workers=${WORKERS}")
    [[ -n "${STOP_HOURS:-}"     ]] && overrides+=("optimization.stop_time_hours=${STOP_HOURS}")

    # ★ 步数与 LR 调度必须一起改，否则会静默欠训练。
    #   fast 配置里 `warmup_steps: 2000` / `decay_steps: 6000` 是按 max_update=8000
    #   写死的，而 tri_stage 用的是**固定步数**（不是 phase_ratio 比例）。
    #   只覆盖 max_update 的后果：例如按 results/02 建议的 MAX_UPDATE=30000，
    #   decay 在第 8000 步就走完了，剩下的 22000 步（73% 的训练时间）全跑在
    #   lr*final_lr_scale = 0.0001 上——日志里看起来"训了 30000 步"，实际上是白跑。
    #   这里沿用 fast 配置原有的 warmup:decay = 1:3（warmup 占 1/4）自动跟随；
    #   想自己定就设 WARMUP_STEPS / DECAY_STEPS，或在 EXTRA 里显式给出 lr_scheduler.*。
    if [[ -n "${MAX_UPDATE:-}" ]] && [[ "${EXTRA:-}" != *warmup_steps* ]]; then
      _wu="${WARMUP_STEPS:-$(( MAX_UPDATE / 4 ))}"
      # 先把实际采用的 decay 算出来再打印，避免"日志里印的和真正追加的覆盖项不一致"
      # （显式给了 DECAY_STEPS 时原来就会不一致——本项目把日志当证据用，不能这样）
      _dec="${DECAY_STEPS:-$(( MAX_UPDATE - _wu ))}"
      overrides+=("lr_scheduler.warmup_steps=${_wu}")
      overrides+=("lr_scheduler.decay_steps=${_dec}")
      echo "LR 调度随 MAX_UPDATE=${MAX_UPDATE} 调整为 warmup=${_wu} / decay=${_dec}"
    fi
    if [[ -n "${EXTRA:-}" ]]; then
      # EXTRA 的约定是"空格分隔的 hydra 覆盖项"，所以需要词分割——但**不需要**
      # 路径名展开：脚本前面已经 `cd "${REPO_ROOT}"`，一旦某个值里带 `*`/`?`
      # 又恰好匹配到仓库根目录下的文件，就会被替换成文件名，hydra 拿到一堆垃圾参数。
      # 临时关掉 glob（set -f）是最省事且不改语义的写法。
      set -f
      # shellcheck disable=SC2206
      overrides+=(${EXTRA})
      set +f
    fi

    if [[ ${#overrides[@]} -gt 0 ]]; then
      echo "命令行覆盖：${overrides[*]}"
    fi

    # 注意：${arr[@]+"${arr[@]}"} 是为了兼容 bash < 4.4 下 set -u + 空数组的情况
    CFG_NAME="$(gen_config base_lrs2_video_only_fast)"
    python fairseq/fairseq_cli/hydra_train.py \
      --config-dir avhubert/conf --config-name "${CFG_NAME}" \
      common.user_dir="${USER_DIR}" \
      distributed_training.distributed_world_size="${NGPU}" \
      distributed_training.nprocs_per_node="${NGPU}" \
      task.data="${DATA}" \
      task.label_dir="${DATA}" \
      task.tokenizer_bpe_model="${BPE}" \
      model.w2v_path="${CKPT}" \
      ${overrides[@]+"${overrides[@]}"} \
      hydra.run.dir="${EXP}/fast"
    echo "快速微调完成。checkpoint 在 ${EXP}/fast"
    ;;

  test)
    FINETUNED="${2:-}"
    [[ -n "${FINETUNED}" ]] || { echo "用法: bash scripts/run_finetune.sh test <微调后的 checkpoint> [gen_subset]"; exit 1; }
    [[ -f "${FINETUNED}" ]] || { echo "找不到 checkpoint: ${FINETUNED}"; exit 1; }
    GEN="${3:-test}"
    OUT="${EXP}/decode_${GEN}"
    # beam 是解码耗时的大头：官方 50，降到 5~10 可省 5~10 倍解码时间，
    # WER 通常只差零点几个点。汇报时务必写清用的 beam。
    BEAM="${BEAM:-50}"
    echo "==== 在 ${GEN} 集上做纯视觉唇读解码（override.modalities=['video'], beam=${BEAM}）===="
    # 官方解码入口是 avhubert/infer_s2s.py（注意：没有 hydra_generate.py）
    #
    # 两个坑（都已实测踩过）：
    # 1) **不要**传 task.data / task.label_dir / task.tokenizer_bpe_model。
    #    infer_s2s.py:112 的 task 配置是**从 checkpoint 里读出来的**
    #    （load_model_ensemble_and_task 返回 saved_cfg.task），s2s_decode.yaml
    #    里根本没有 task 段，传 task.* 会直接报
    #        Could not override 'task.data'. Key 'data' is not in struct
    #    想换数据目录要用 override.data / override.label_dir（infer_s2s.py:136-139）。
    # 2) WER 文件名不是 wer.${GEN}，而是 **wer.<generation 配置的哈希>**
    #    （infer_s2s.py:247-259），所以下面用通配符去找。
    python avhubert/infer_s2s.py \
      --config-dir avhubert/conf --config-name s2s_decode \
      common.user_dir="${USER_DIR}" \
      dataset.gen_subset="${GEN}" \
      common_eval.path="${FINETUNED}" \
      common_eval.results_path="${OUT}" \
      generation.beam="${BEAM}" \
      override.modalities="['video']" \
      override.data="${DATA}" \
      override.label_dir="${DATA}" \
      hydra.run.dir="${OUT}"
    echo
    echo "================ 测试结果 ================"
    # 注意：文件名是 wer.<哈希>，不是 wer.${GEN}，所以要通配查找
    shopt -s nullglob
    wer_files=("${OUT}"/wer.*)
    shopt -u nullglob
    if [[ ${#wer_files[@]} -gt 0 ]]; then
      for f in "${wer_files[@]}"; do
        echo "--- $(basename "$f") ---"
        head -2 "$f"
      done
      echo "（这就是题目要求汇报的『点数』：LRS2 ${GEN} 集、纯视觉模态的 WER）"
    else
      echo "未找到 ${OUT}/wer.*，请检查上面的日志输出。"
    fi
    echo "详细结果：${OUT}（hypo-<哈希>.json 为逐句假设）"
    ;;

  *)
    echo "用法: bash scripts/run_finetune.sh [smoke|train|fast|test] [checkpoint] [gen_subset]"
    exit 1
    ;;
esac
