#!/usr/bin/env bash
# =============================================================================
# pipeline_lrs2.sh -- AV-HuBERT 在 LRS2 上复现的完整数据流水线
#
# 覆盖：数据列表 -> （抽音频）-> 人脸关键点 -> 唇部 ROI -> 数帧 -> manifest/词表
# 前提：已在 Linux + conda 环境（见 environment.yml）中，且已激活该环境。
#
# 用法：
#   export LRS2_ROOT=/path/to/lrs2/main          # 原始数据集（含 train/val/test）
#   export DATALIST=/path/to/lrs2_datalist       # 招新题目给的列表目录
#   export WORK=/path/to/lrs2_data               # 工作目录（输出都放这里）
#   export FFMPEG=$(which ffmpeg)
#   export DLIB=/path/to/dlib_models             # dlib 模型目录
#   export NSHARD=8                              # 并行分片数
#   export MODALITY=video                        # video（默认，纯视觉）| av（音视频）
#   export WARP_BACKEND=cv2                      # cv2（默认，快 4.4x）| skimage（官方）
#   export WRITE_BACKEND=pipe                    # pipe（默认）| png（官方逐帧 PNG）
#   bash scripts/pipeline_lrs2.sh all
#
# 也可单独执行某一步：
#   bash scripts/pipeline_lrs2.sh prepare|audio|landmark|mouth|frames|manifest
#
# 关于 WARP_BACKEND / WRITE_BACKEND：
#   默认的 cv2 + pipe 是经过逐位校验的快路径（解码前 ROI 帧平均差 0.5 灰度级，
#   经 h264 后产物层面平均差 2.26 灰度级 ≈ 0.9%，与压缩噪声同量级）。
#   若要产出与官方预处理**逐位可比**的 ROI，设
#   WARP_BACKEND=skimage WRITE_BACKEND=png（这一阶段会从 37 分钟回到 1.5 小时以上）。
#
# 关于 MODALITY（很重要，直接关系耗时）：
#   本题要求「仅使用视觉模态」。纯视觉训练**不会读取音频 wav**
#   （hubert_dataset.py:283,359 只在 'audio' in modalities 时读；
#     tsv 的音频列只当字符串用来取 utterance id，音频帧数列从未被读），
#   所以 MODALITY=video 会跳过整步抽音频，省约 1 小时 + 3.3 GB，
#   并让 count_frames 用 --video-only（不读 wav，该列写 0 占位）。
#   要做音视频联合训练就设 MODALITY=av，行为与官方一致。
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
# NSHARD 必须是正整数：`:=` 只在"未设置/为空"时赋值，所以 NSHARD=0 会原样留下，
# 于是 run_ranks_parallel 启动 **0 个 rank**、失败数 0，第 2/3/4 步"全部成功"
# 却什么都没做——真正的报错要到第 6 步才以"长度不一致"的形式出现，极难定位。
if ! [[ "${NSHARD}" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：NSHARD 必须是正整数（当前：'${NSHARD}'）。分片数应为 CPU 配额大小，例如 NSHARD=8。" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# 模态：video（默认） | av
#   本题要求「仅使用视觉模态」，而抽音频这一步对纯视觉训练**完全用不到**：
#     - hubert_dataset.py:283,359 只在 'audio' in modalities 时读 wav；
#     - tsv 里音频那一列只被当作字符串用来取 utterance id（items[2]），
#       音频帧数列（items[-1]）从头到尾没有被读过。
#   所以 MODALITY=video 时会跳过整步抽音频（省约 1 小时 + 3.3 GB），
#   并让 count_frames.py 用 --video-only（不读 wav，该列写 0 占位）。
#   需要音视频联合训练时才用 MODALITY=av，行为与官方一致。
# ---------------------------------------------------------------------------
: "${MODALITY:=video}"
case "${MODALITY}" in
  video) VIDEO_ONLY_ARG=(--video-only) ;;
  av)    VIDEO_ONLY_ARG=() ;;
  *)     echo "错误：MODALITY 只能是 video 或 av（当前：${MODALITY}）" >&2; exit 2 ;;
esac

# 唇部 ROI 的两个后端开关。默认是经过实测校验的快路径；设成
#   WARP_BACKEND=skimage WRITE_BACKEND=png
# 即可完全退回官方实现（产物与官方预处理逐位可比），无需改脚本。
# 默认值与 align_mouth.py 里 argparse 的默认值保持一致。
: "${WARP_BACKEND:=cv2}"
: "${WRITE_BACKEND:=pipe}"
case "${WARP_BACKEND}" in
  cv2|skimage) ;;
  *) echo "错误：WARP_BACKEND 只能是 cv2 或 skimage（当前：${WARP_BACKEND}）" >&2; exit 2 ;;
esac
case "${WRITE_BACKEND}" in
  pipe|png) ;;
  *) echo "错误：WRITE_BACKEND 只能是 pipe 或 png（当前：${WRITE_BACKEND}）" >&2; exit 2 ;;
esac

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
  # 注意结尾的 `|| true`：本脚本是 `set -euo pipefail`，df 一旦失败
  # （老版本 coreutils 不认 `--output`、路径不存在等）赋值就是非 0 状态，
  # 整条流水线会**在这里直接中止**，下面那句"跳过检查"的优雅降级永远不会执行
  # （同理还有后面的 ${hy_gb:-未知}）。
  avail_gb=$(df -BG --output=avail "${target}" 2>/dev/null | tail -1 | tr -dc '0-9') || true
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
  hy_gb=$(df -BG --output=avail /hy-tmp 2>/dev/null | tail -1 | tr -dc '0-9') || true
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
    ${limit_args[@]+"${limit_args[@]}"}
  echo "总条数: $(wc -l < "${WORK}/file.list")"
  echo "fid 样例: $(head -1 "${WORK}/file.list")"
}

# ---------- 2. 抽取音频（wav, 16kHz 单声道）----------
# 注意：原始 LRS2 音频就是 16kHz。这里统一转成 wav 以匹配
#       hubert_dataset 里 scipy.io.wavfile 的读取方式。
step_audio() {
  local rank="${1:-0}"
  if [[ "${MODALITY}" == "video" ]]; then
    # 纯视觉：这一步的产物不会被任何后续步骤读取（见文件头 MODALITY 的说明），
    # 直接跳过，省下约 1 小时和 3.3 GB。需要一个"音频路径字符串"来当
    # utterance id，那是 lrs2_manifest.py 拼出来的，与文件是否存在无关。
    log "step 2/6  跳过抽音频（MODALITY=video，纯视觉训练不读音频）"
    return 0
  fi
  log "step 2/6  从 mp4 抽取音频到 ${WORK}/audio"
  python - "$LRS2_ROOT" "$WORK" "$FFMPEG" "$NSHARD" "$rank" <<'PY'
import os, sys, math, subprocess
from tqdm import tqdm

lrs2_root, work, ffmpeg, nshard, rank = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
fids = [ln.strip() for ln in open(os.path.join(work, 'file.list')) if ln.strip()]
n = math.ceil(len(fids) / nshard)
shard = fids[rank*n:(rank+1)*n]
print(f"rank {rank}/{nshard}: {len(shard)} clips")
failed = []
for fid in tqdm(shard):
    # fid 与实际目录结构一致（flat 布局下即 <video_id>/<clip_id>），可直接拼路径
    src = os.path.join(lrs2_root, fid + '.mp4')
    dst = os.path.join(work, 'audio', fid + '.wav')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(dst):
        continue
    # 检查返回码：原实现丢掉返回值，转码失败只会表现为"文件不存在"，
    # 而且要等到第 5 步数帧时才发现（中间还隔着最慢的关键点阶段）。
    rc = subprocess.call([ffmpeg, '-loglevel', 'quiet', '-i', src, '-f', 'wav',
                          '-ar', '16000', '-ac', '1', '-vn', '-y', dst])
    if rc != 0 or not os.path.isfile(dst):
        if os.path.isfile(dst):
            os.remove(dst)          # 别留下半截 wav 被"已存在就跳过"永久信任
        failed.append((fid, rc))
if failed:
    print("rank %d: %d 个 clip 抽音频失败：" % (rank, len(failed)))
    for fid, rc in failed[:10]:
        print("   %s (ffmpeg rc=%s)" % (fid, rc))
    sys.exit(1)
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
  log "step 4/6  裁切嘴部 ROI -> ${WORK}/video（warp=${WARP_BACKEND}, write=${WRITE_BACKEND}）"
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
    --warp-backend  "${WARP_BACKEND}" \
    --write-backend "${WRITE_BACKEND}" \
    --rank          "${rank}" \
    --nshard        "${NSHARD}"
}

# ---------- 5. 统计帧数并合并分片 ----------
step_frames() {
  log "step 5/6  统计帧数（MODALITY=${MODALITY}）"
  # count_frames.py 只在"发现有缺失"时才写 missing.list.<rank>，全仓库没有别处删它。
  # 残留会让本步**永远**失败（compgen 匹配到旧文件），而且要清掉它往往得
  # rm -rf 整个 WORK —— 那会把没有续跑逻辑、最慢的关键点阶段一起丢掉。
  # 所以每轮开始先清干净。
  rm -f "${WORK}"/missing.list.*
  # 同理清掉上一轮的 per-rank 帧数文件：count_frames.py 是直接写最终文件名的，
  # 某个 rank 在写之前被 kill 就会留下**上一轮**的 nframes.*.<rank>，
  # 下面的存在性检查照样通过，于是把过期帧数并进 nframes.*（长度对、内容错，
  # 后续 lrs2_manifest 的长度交叉校验抓不到）。
  rm -f "${WORK}"/nframes.audio.* "${WORK}"/nframes.video.*
  for ((r=0; r<NSHARD; r++)); do
    python "${PREP}/count_frames.py" \
      --root     "${WORK}" \
      --manifest "${WORK}/file.list" \
      --nshard   "${NSHARD}" \
      --rank     "$r" \
      ${VIDEO_ONLY_ARG[@]+"${VIDEO_ONLY_ARG[@]}"}
  done
  # 先检查缺失、再合并：原顺序是先 cat 再检查，而某个 rank 有缺失时不会写
  # nframes.*.<rank>，set -e 下 cat 会先以"No such file"失败退出，
  # 于是那条"存在缺失文件清单"的人话提示永远打不出来。
  if compgen -G "${WORK}/missing.list.*" > /dev/null; then
    echo "错误：存在缺失/不可解码文件清单，请检查：" >&2
    cat "${WORK}"/missing.list.* >&2
    exit 1
  fi
  : > "${WORK}/nframes.audio"
  : > "${WORK}/nframes.video"
  for ((r=0; r<NSHARD; r++)); do
    [[ -f "${WORK}/nframes.audio.${r}" ]] || { echo "错误：缺少 ${WORK}/nframes.audio.${r}" >&2; exit 1; }
    [[ -f "${WORK}/nframes.video.${r}" ]] || { echo "错误：缺少 ${WORK}/nframes.video.${r}" >&2; exit 1; }
    cat "${WORK}/nframes.audio.${r}" >> "${WORK}/nframes.audio"
    cat "${WORK}/nframes.video.${r}" >> "${WORK}/nframes.video"
  done
  echo "音频帧数行数: $(wc -l < "${WORK}/nframes.audio")"
  echo "视频帧数行数: $(wc -l < "${WORK}/nframes.video")"
  if [[ "${MODALITY}" == "video" ]]; then
    echo "（纯视觉模式：nframes.audio 全是 0，这是占位列，训练不会读它）"
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

# ---------- 7. 全量完整性验收 ----------
# scripts/verify_dataset.py 校验的是 manifest 的**核心契约**：
#   tsv 里的视频帧数 == landmark 条数 == ROI 实际帧数
# 这三者一旦不一致，训练**不会报错**，只会静默用错长度的样本或错的分批。
# 这个脚本原来只被文档提到、没有任何脚本调用它，所以"一键复跑"里缺了这道验收。
step_verify() {
  log "step 7/7  全量完整性验收（verify_dataset.py）"
  [[ -f "${WORK}/nframes.video" ]] || {
    echo "跳过验收：${WORK}/nframes.video 不存在（先跑 frames 步骤）" >&2; return 0; }
  local vargs=()
  [[ -n "${VERIFY_LIMIT:-}" ]] && vargs=(--limit "${VERIFY_LIMIT}")
  # 用绝对路径：本脚本不依赖 cwd（其余步骤也都用 ${PREP} 绝对路径）
  python "${REPO_ROOT}/scripts/verify_dataset.py" \
    --work-dir "${WORK}" \
    --workers  "${NSHARD}" \
    ${vargs[@]+"${vargs[@]}"}
}

# ---------- 并行调度：同一阶段的所有 rank 同时跑，并逐个检查退出码 ----------
# 用法: run_ranks_parallel <step_fn_name>
# 说明:
#   - 不用无参数的 `wait`：那样无法区分是哪个 rank 失败，也拿不到退出码。
#   - 任一 rank 失败即 exit 1，避免带着残缺产物继续跑后面的阶段
#     （否则问题要到几小时后才暴露）。
run_ranks_parallel() {
  local step_fn="$1"
  local pids=() failed=0

  for ((r=0; r<NSHARD; r++)); do
    "${step_fn}" "$r" &
    pids+=($!)
  done

  for ((i=0; i<${#pids[@]}; i++)); do
    if ! wait "${pids[$i]}"; then
      echo "错误：${step_fn} rank $i 失败 (pid ${pids[$i]})" >&2
      failed=1
    fi
  done

  if [[ "${failed}" -ne 0 ]]; then
    echo "错误：${step_fn} 存在失败的 rank，终止流水线。" >&2
    exit 1
  fi
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

    # 阶段级并行：每个阶段内 NSHARD 个 rank 同时跑，跑完再进下一阶段。
    # 阶段之间必须保持顺序（step_frames 依赖全部产物就绪）。
    if [[ "${MODALITY}" == "video" ]]; then
      log "step 2/6  跳过抽音频（MODALITY=video，纯视觉训练不读音频；要音视频请设 MODALITY=av）"
    else
      log "step 2/6  并行抽取音频（${NSHARD} 个 rank）"
    fi
    run_ranks_parallel step_audio

    log "step 3/6  并行检测人脸关键点（${NSHARD} 个 rank）"
    run_ranks_parallel step_landmark

    # 清掉上一次被 kill 时残留的原子发布中间文件（align_mouth.py 写 `<fid>.mp4.tmp.mp4`
    # 再用 os.replace 改名）。**必须在这里清、不能在 step_mouth 里清**：step_mouth 是
    # 并行 rank，在函数内清会删掉别的 rank 正在写的临时文件。这里还没启动任何 rank，安全。
    # ⚠ 必须用 find 而不是 `rm -f "${WORK}"/video/*.tmp.mp4`：
    #   fid 形如 `<video_id>/<clip_id>`，所以 ROI 是**嵌套**目录
    #   （`video/5535415699068794046/00001.mp4`），而 shell 的 `*` 不跨 `/`——
    #   那样写一条都匹配不到，等于死代码（已实测：2 个残留文件纹丝不动）。
    find "${WORK}/video" -type f -name '*.tmp.mp4' -delete 2>/dev/null || true

    log "step 4/6  并行裁切唇部 ROI（${NSHARD} 个 rank）"
    run_ranks_parallel step_mouth

    step_frames
    step_manifest
    step_verify
    ;;
  verify)   step_verify ;;
  *)
    echo "用法: bash scripts/pipeline_lrs2.sh [all|prepare|audio|landmark|mouth|frames|manifest|verify] [rank]"
    exit 1
    ;;
esac

log "完成。数据目录：${WORK}/data"
