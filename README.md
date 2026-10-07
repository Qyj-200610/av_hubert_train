# AV-HuBERT 复现：在 LRS2 上的纯视觉唇读测试

复现 [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert)，
使用其预训练权重，在 **LRS2 测试集** 上做 **仅视觉模态** 的唇读测试，汇报 WER（点数）。

> 官方的原始 README 已保留为 [`README_upstream.md`](README_upstream.md)。

---

## 1. 当前进度

| 项目 | 状态 |
| --- | --- |
| 官方代码导入（av_hubert + fairseq 子模块，固定 commit） | ✅ 已完成 |
| 预训练权重下载（`base_vox_iter4.pt`，1.18 GB） | ✅ 已完成 |
| LRS2 数据预处理脚本（`lrs2_prepare.py`） | ✅ 已完成 |
| LRS2 manifest / 词表生成（`lrs2_manifest.py`） | ✅ 已完成 |
| LRS2 finetune 配置（video-only：base + smoke） | ✅ 已完成 |
| 环境规格（`environment.yml`）与一键脚本 | ✅ 已完成 |
| LRS2 数据集下载（约 50 GB） | ⬜ 待办 |
| conda 环境创建 + fairseq 编译 | ⬜ 待办（需 Linux 服务器 / WSL2） |
| 微调 + 测试 + 汇报 WER | ⬜ 待办 |

> 本机（Windows + Python 3.12 + RTX 5060 Laptop 8GB）**不适合**跑这个实验：
> fairseq 的 pinned commit 只支持 Python 3.8–3.10，且需在 Linux 上编译 C++ 扩展。
> 因此本地只完成「代码 + 配置 + 权重」骨架，训练/测试请在 Linux 服务器或 WSL2 上执行。

---

## 2. 版本信息（复现可追溯）

| 组件 | 版本 / commit |
| --- | --- |
| av_hubert | `258fb50e155134eec2c4b49c2ae8de267075fd18`（2023-07-09，「update scipy version due to security vulnerability」） |
| fairseq（子模块） | `afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb`（2021-06-15） |
| 预训练权重 | `base_vox_iter4.pt`，来源 <https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter4.pt> |
| 数据集 | LRS2 **main**（train / val / test 三部分） |

全部官方 checkpoint 见 <https://facebookresearch.github.io/av_hubert/>。备选权重：

| 权重 | 规模 | 说明 |
| --- | --- | --- |
| `base_vox_iter4.pt` | 约 1.18 GB | **本项目默认**。base 模型 + VoxCeleb2 干净预训练，显存占用小，易于微调 |
| `large_vox_iter5.pt` | 约 3 GB | large 模型，效果更好但显存与时间开销显著增加 |
| `base_noise_pt_noise_ft_30h.pt` / `large_noise_pt_noise_ft_433h.pt` | — | **已在 LRS3 上微调过**（带噪声增强），可走「用官方权重直接测试」的快速路线 |

---

## 3. 目录结构

```
av_hubert_train/
├── avhubert/                     # 官方核心代码（含本项目新增的 LRS2 适配）
│   ├── conf/
│   │   ├── finetune/
│   │   │   ├── base_lrs2_video_only.yaml   # ★ 新增：LRS2 纯视觉正式微调配置
│   │   │   ├── smoke_lrs2.yaml             # ★ 新增：小规模冒烟测试配置
│   │   │   └── base_vox_433h.yaml ...      # 官方原有（LRS3/Vox）
│   │   ├── pretrain/                       # 官方预训练配置
│   │   └── s2s_decode.yaml                 # 官方解码配置
│   ├── preparation/
│   │   ├── lrs2_prepare.py                 # ★ 新增：LRS2 数据准备
│   │   ├── lrs2_manifest.py                # ★ 新增：LRS2 manifest / 词表生成
│   │   ├── detect_landmark.py              # 官方（复用）
│   │   ├── align_mouth.py                  # 官方（复用）
│   │   ├── count_frames.py                 # 官方（复用）
│   │   └── gen_subword.py                  # 官方（复用，生成 sentencepiece 词表）
│   ├── hubert_asr.py                       # finetune 模型（av_hubert_seq2seq）
│   ├── hubert_dataset.py                   # 数据加载（决定 tsv 格式）
│   ├── hubert_pretraining.py               # task 定义（决定 tsv/wrd/dict 命名）
│   └── infer_s2s.py                        # ★ 解码入口（注意：官方没有 hydra_generate.py）
├── fairseq/                      # 官方子模块（pinned commit）
├── pretrained/
│   └── base_vox_iter4.pt         # 已下载的预训练权重
├── scripts/                      # ★ 新增：本项目脚本
│   ├── pipeline_lrs2.sh          # 数据流水线（6 步）
│   ├── run_finetune.sh           # 冒烟 / 微调 / 测试
│   └── fetch_dlib_models.sh      # 下载 dlib 模型与参考平均脸
├── environment.yml               # ★ 新增：conda 环境（Linux + GPU）
├── README_upstream.md            # 官方原始 README
└── README.md                     # 本文件
```

★ = 本项目新增或修改的文件。

---

## 4. 环境搭建（Linux / WSL2）

```bash
conda env create -f environment.yml
conda activate avhubert
bash scripts/setup_env.sh          # 编译 fairseq 扩展 + 安装 avhubert
```

**版本选择的关键原因（踩坑点）：**

1. `fairseq` 被 pin 在 2021-06 的 commit，**只兼容 Python 3.8–3.10**，且必须 **numpy 1.x**
   （numpy 2.x 会导致 C++ 扩展编译失败）。`environment.yml` 固定 `python=3.9` + `numpy=1.24.4`。
2. `av_hubert/requirements.txt` 中的 `opencv-python==4.5.4.60` 没有 py3.9 的 wheel，
   已改为 `4.9.0.80`（API 兼容）。
3. `torch 1.13.1` + `cudatoolkit 11.7` 是该 commit 时代最好装的组合；
   若服务器 CUDA 更新（如 12.x），conda 环境内的 11.7 运行时依然可用（驱动够新即可）。

---

## 5. 数据准备

### 5.1 与 LRS3 的差异（务必先读）

官方只提供 LRS3 / VoxCeleb2 的预处理脚本，直接套用会踩三个坑：

| 差异 | 说明 | 处理 |
| --- | --- | --- |
| LRS2 无需切句 | main 里每个 clip 已是切好的短句（≤ 约 15 s）并自带标注 | 跳过官方 LRS3 的 step 1/2（按时间戳切长视频） |
| **没有 pretrain 划分** | 题目 `Dataset.md` 已提示；官方 433h 训练集依赖 pretrain | 只用 train/val/test，不再区分 30h / 433h |
| **划分目录名是 `val` 而非 `valid`** | 官方 `detect_landmark.py` / `align_mouth.py` 内部按 `{root}/{fid}.mp4` 拼路径 | `file.list` 前缀**必须**用真实目录名 `train/val/test`；fairseq 需要的 `valid.tsv` 由 `lrs2_manifest.py` 改名产出 |

### 5.2 数据划分（题目提供）

`任务要求/lrs2_datalist/`：

| 文件 | 行数 | 格式 |
| --- | --- | --- |
| `train.txt` | 45,839 | 相对路径，无标签 |
| `val.txt` | 1,082 | 相对路径，无标签 |
| `test.txt` | 1,243 | **路径 + `NF`/`MV` 标记** 两列 |

转写文本来自 LRS2 数据集自带的逐句 `.txt` 文件，由 `lrs2_prepare.py` 读取。

### 5.3 数据格式契约（摘自官方代码，改数据时必须遵守）

- `hubert_pretraining.py` → `load_dataset()`：读 `{task.data}/{split}.tsv`；
  标签读 `{task.label_dir}/{split}.wrd`；词典读 `{label_dir}/dict.wrd.txt`。
- `hubert_dataset.py` → `load_audio_visual()`：tsv **首行是 root**（被跳过），
  其余每行以 `\t` 分隔：`items[0]=id`、`items[1]=video`、`items[2]=audio`、
  **`items[-2]` = 视频帧数**、`items[-1]` = 音频帧数。
- 帧率 25 fps；`max_keep` 默认 500 帧（即 20 s）。
- `dict.wrd.txt` 就是 sentencepiece 导出的 `spm_unigram{N}.txt`（fairseq Dictionary 格式）。

### 5.4 执行流水线

```bash
export LRS2_ROOT=/path/to/lrs2/main          # 含 train/val/test 三个子目录
export DATALIST=/path/to/lrs2_datalist
export WORK=/path/to/lrs2_data               # 输出都放这里
export FFMPEG=$(which ffmpeg)
export DLIB=/path/to/dlib
export NSHARD=8                              # 并行分片数

bash scripts/fetch_dlib_models.sh "$DLIB"    # 下载 dlib 模型 + 20words_mean_face.npy
bash scripts/pipeline_lrs2.sh all
```

流水线共 6 步：

1. `lrs2_prepare.py` —— 解析三个列表 → `file.list` / `label.list`（严格校验缺标注 / 缺 mp4）
2. 抽音频 —— ffmpeg 转 16 kHz 单声道 wav 到 `$WORK/audio/`
3. `detect_landmark.py` —— dlib 68 点关键点 → `$WORK/landmark/`
4. `align_mouth.py` —— 相似变换对齐 + 裁 96×96 嘴部 ROI → `$WORK/video/`
5. `count_frames.py` —— 统计帧数 → `nframes.audio` / `nframes.video`
6. `lrs2_manifest.py` —— 生成 `{train,valid,test}.{tsv,wrd}` + `dict.wrd.txt` + spm 词表

产出位于 `$WORK/data/`。**注意 `nframes.*` 必须与 `file.list` 顺序严格一致**
（`count_frames.py` 按分片写出，需按 rank 顺序拼接，流水线已处理）。

---

## 6. 微调与测试

```bash
export DATA=$WORK/data
export CKPT=$(pwd)/pretrained/base_vox_iter4.pt
export EXP=$(pwd)/exp

# 1) 先冒烟测试：20 步，确认数据 / 模型 / 反传都通
bash scripts/run_finetune.sh smoke

# 2) 正式微调（8 卡；单卡会自动把 world_size 设为 1）
NGPU=8 bash scripts/run_finetune.sh train

# 3) 在 test 集上做纯视觉唇读解码，输出 WER
bash scripts/run_finetune.sh test "$EXP/train/checkpoint_best.pt" test
```

### 关键配置说明

`avhubert/conf/finetune/base_lrs2_video_only.yaml`：

- `task.modalities: ["video"]` —— **纯视觉**，不使用音频（题目硬性要求）
- `task.normalize: true` —— 必须与预训练一致。
  已核对 `base_vox_iter4.pt` 内的 cfg：`task.normalize = True`，一致 ✅
  （`hubert_asr.py` 中有 assert，不一致会直接报错）
- `model.w2v_path` —— 预训练权重；模型会从 checkpoint 内部读取预训练的 `cfg.model`
  （含 `model.label_rate=25` 等），因此**不需要**额外的预训练配置
- `dataset.valid_subset: valid` —— 对应 `valid.tsv`（即 LRS2 的 val 划分）
- `max_update: 30000`、`freeze_finetune_updates: 30000` —— 沿用官方 433h 设置

### 解码（官方命令，注意入口名）

官方**没有** `fairseq_cli/hydra_generate.py`，解码入口是 `avhubert/infer_s2s.py`：

```bash
python avhubert/infer_s2s.py --config-dir avhubert/conf --config-name s2s_decode \
  dataset.gen_subset=test common_eval.path=/path/to/checkpoint \
  common_eval.results_path=/path/to/decode \
  override.modalities="['video']" common.user_dir=$(pwd)
```

结果写入 `{results_path}/wer.{split}`，形如 `WER: 12.34`。**这就是题目要汇报的「点数」。**

---

## 7. 评测口径（重要，建议两个都报）

`test.txt` 的 1,243 条自带 `NF` / `MV` 标记（来自 LRS2 的 pretrain 部分）。
`lrs2_manifest.py` 已额外生成 `test_mv.{tsv,wrd}`，可直接：

```bash
bash scripts/run_finetune.sh test "$EXP/train/checkpoint_best.pt" test_mv
```

建议同时汇报：

- **全量 1,243 条**（主口径）
- **仅 `MV`（mouth-visible）子集**（LRS2 论文常用的干净口径）

并写清所用预训练权重档次、微调步数、解码 beam size。

---

## 8. 已知风险与后续事项

1. **算力**：LRS2 main 训练集约 100+ 小时音频量，`base` 模型 8 卡微调约需数天。
   算力有限时可降低 `optimization.max_update`（如 10000），或先用 `--max-train` 跑子集。
2. **本机无法训练**：Windows + Python 3.12 装不了 fairseq（pinned commit 需 py ≤ 3.10 + Linux 编译）；
   本机 RTX 5060 Laptop（8 GB）显存也不足以运行 av_hubert。
3. **`test.txt` 的标签列**不是转写文本，只是 `NF` / `MV` 标记，转写需从数据集的 `.txt` 取得。
4. **`val` vs `valid` 命名**：自行改脚本时务必保持「fid 前缀 = 真实目录名」，
   否则 `detect_landmark.py` / `align_mouth.py` 会报 `File does not exist`。
5. **帧数文件顺序**：`nframes.audio` / `nframes.video` 必须与 `file.list` 逐行对应。
6. 官方 checkpoint 全部基于 **LRS3 / VoxCeleb2**（英文），LRS2 同为英文，可直接微调。

---

## 9. 参考

- 论文：*Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction*，<https://arxiv.org/abs/2201.02184>
- 代码：<https://github.com/facebookresearch/av_hubert>
- 权重：<https://facebookresearch.github.io/av_hubert/>
- LRS2 数据：<https://aistudio.baidu.com/datasetdetail/132643/0>
- 官方预处理说明：`avhubert/preparation/README.md`
