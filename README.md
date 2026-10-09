# AV-HuBERT 复现：在 LRS2 上的纯视觉唇读测试

复现 [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert)，
使用其预训练权重，在 **LRS2 测试集** 上做 **仅视觉模态** 的唇读测试，汇报 WER（点数）。

> 官方的原始 README 已保留为 [`README_upstream.md`](README_upstream.md)。

---

## ★ 复现结果（LRS2 test，纯视觉）

| 评测口径 | 样本数 | beam | **WER** |
| --- | --- | --- | --- |
| **LRS2 `test`（全量，主口径）** | **1,243** | 10 | **52.97%** |
| LRS2 `test_mv`（MV 子集） | 246 | 10 | **56.96%** |

- 配置：官方 `base_vox_iter4.pt` + 纯视觉（`modalities: ["video"]`）、8,000 步微调、
  编码器全程冻结、beam=10；单卡 RTX 3090 实测 **44.3 分钟**（0.33 秒/步）。
- 全部结果、原始日志、WER 文件与逐句假设见 **[`results/`](results/)**；
  任务要求格式的结果记录见 [`docs/复现结果.md`](docs/复现结果.md)。
- 逐句层面：0 条空输出、14.6% 的句子完全正确、逐句中位 WER 50%（说明模型确实在做唇读）。
- ⚠ 该结果使用的训练步数与编码器冻结策略都**比官方配方更省时**（官方为 45,000 步 +
  后半程解冻编码器），因此 52.97% 应理解为"算力受限下的结果"，不是方法上限；
  提升路径见 `results/02_微调与解码/配置与结果.md` §6。

## 1. 当前进度

| 项目 | 状态 |
| --- | --- |
| 官方代码导入（av_hubert + fairseq） | ✅ 已完成（注：`fairseq/` 是**普通提交的目录树**，不是 git submodule——`git ls-tree HEAD fairseq` 返回 tree 而非 commit，`.gitmodules` 里的声明在交付仓库里无法校验） |
| 预训练权重下载（`base_vox_iter4.pt`，1.18 GB） | ✅ 已完成 |
| LRS2 数据预处理脚本（`lrs2_prepare.py`） | ✅ 已完成 |
| LRS2 manifest / 词表生成（`lrs2_manifest.py`） | ✅ 已完成 |
| LRS2 finetune 配置（video-only：base + smoke） | ✅ 已完成 |
| 环境规格（`environment.yml`）与一键脚本 | ✅ 已完成 |
| LRS2 **main** 数据集下载（用题目给的地址，勿下完整版 LRS2） | 已完成（48,164 条 clip） |
| 租云 GPU + 创建 conda 环境 + 编译 fairseq | 已完成（恒源云 1×RTX 3090） |
| 微调 + 测试 + 汇报 WER | 已完成：**LRS2 test 纯视觉 WER = 52.97%**（1,243 条，beam=10）；MV 子集 56.96%（246 条）。结果与原始产物见 `results/` |

> **本机跑不了这个实验**（已实测，三条硬约束）：Windows + Python 3.12 装不了 fairseq
> （新版 Python 拒绝其 dataclass 写法）；本机 RTX 5060 Laptop 是 **sm_120（Blackwell）**，
> 而 fairseq 配套的 `torch 1.13.1+cu117` 最高只支持 sm_86；8 GB 显存也远不够微调。
> 因此**推荐路线是把实验放到租用的云 GPU 上做**，详见
> [4.1 为什么不在本机跑（推荐路线：租云 GPU）](#41-为什么不在本机跑推荐路线租云-gpu)。
> 本机只负责代码维护与 `git` 版本管理。

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
│   └── base_vox_iter4.pt         # 已下载的预训练权重（本地保留，云上可重新下载）
├── scripts/                      # ★ 新增：本项目脚本
│   ├── check_gpu_compat.py       # 租到云 GPU 后先跑这个，验证算力 sm_XX 是否被 torch 支持
│   ├── pipeline_lrs2.sh          # 数据流水线（6 步）
│   ├── run_finetune.sh           # 冒烟 / 微调 / 测试
│   ├── setup_env.sh              # 安装依赖 + 编译 fairseq 扩展
│   └── fetch_dlib_models.sh      # 下载 dlib 模型与参考平均脸
├── environment.yml               # ★ 新增：conda 环境（云 GPU 上的 Linux）
├── README_upstream.md            # 官方原始 README
└── README.md                     # 本文件
```

★ = 本项目新增或修改的文件。

---

## 4. 环境搭建（Linux）

> ⚠️ 先看 [4.1 为什么不在本机跑](#41-为什么不在本机跑推荐路线租云-gpu)：
> 本机（Windows + Python 3.12 + RTX 5060 Laptop 8GB）**跑不了**这个实验，
> 环境要建在服务器或租用的云 GPU 上。

```bash
conda env create -f environment.yml
conda activate avhubert
bash scripts/setup_env.sh          # 编译 fairseq 扩展 + 安装 avhubert
```

**版本选择的关键原因（踩坑点）：**

1. `fairseq` 被 pin 在 2021-06 的 commit，**只兼容 Python 3.8–3.10**，且必须 **numpy 1.x**
   （numpy 2.x 会导致 C++ 扩展编译失败）。`environment.yml` 固定 `python=3.8` + `numpy=1.24.4`。
   > 实测证据（同一段 dataclass 代码，两种解释器对比）：
   >
   > | Python | 结果 |
   > | --- | --- |
   > | **3.8.20** | `OK` —— 接受 fairseq 的 `common: CommonConfig = CommonConfig()` 写法 ✅ |
   > | 3.12.10 | `ValueError: mutable default ... for field common is not allowed` ❌ |
   >
   > fairseq 在 `FairseqConfig` 里用了可变 dataclass 默认值，新版 Python 收紧了检查，
   > 这条没有商量余地，**必须 ≤ 3.10**。本仓库代码（含本项目新增的 LRS2 脚本）
   > 已在 Python 3.8 下整体编译通过，未使用任何 3.9+ 专有语法或 API。
2. `av_hubert/requirements.txt` 中的 `opencv-python==4.5.4.60` 没有 py3.8 的 wheel，
   已改为 `4.9.0.80`（API 兼容）。
3. `torch 1.13.1` + `cudatoolkit 11.7` 是该 commit 时代最好装的组合，
   与恒源云镜像可选的「**PyTorch 1.13.1 / 11.7.0 / 3.8**」完全对齐。
   **但只适用于算力 ≤ sm_86 的卡**（如 2080 Ti / V100 / 3090）。
   若你的卡更新，见下节。

### 4.1 为什么不在本机跑（推荐路线：租云 GPU）

本机实测的三条硬约束，**任何一条单独出现都够呛，叠起来就是死路**：

| # | 约束 | 实测证据 | 后果 |
| --- | --- | --- | --- |
| 1 | Python 版本 | Python 3.12 下 `import fairseq` 报 `mutable default ... not allowed` | fairseq 无法导入，需要 ≤ 3.10 |
| 2 | GPU 架构 | 本机 `compute_cap = 12.0`（Blackwell, **sm_120**）；而 `torch 1.13.1+cu117` 的预编译包最高只到 **sm_86** | 老版本 torch 根本不认识这张卡 |
| 3 | 显存与算力 | 8 GB 单张笔记本卡，而微调是「数万步 × ~1.5 GB 权重」 | 量级差几十倍，不是「慢一点」 |

（本机现有 `torch 2.11.0+cu128` 的支持列表含 `sm_120`，所以卡本身没问题；
问题在于 fairseq 那份老代码配不上新 torch/Python。二者只能二选一。）

**因此推荐路线 A：把实验放到租用的云 GPU 上做。**

#### 为什么是租而不是攒

- 任务需要的只是**一次**微调 + 一次测试，不是长期训练平台
- 国内按小时计费的平台（AutoDL 等）租一张 **3090（24 GB，sm_86）或 A100**，
  跑完 `base` 模型的 LRS2 微调大致是「几天、几十元」量级——
  远低于为此买卡，也远低于在笔记本上耗几天到几周
- 注意：**不要贪新选 4090/H100**，它们算力超出 `torch 1.13.1` 的支持范围（见下）

#### 在云上的执行流程

```bash
# ① 传代码（只传代码！数据集和权重都不要传）
#    本机执行：把仓库打包上传，或用 git（前提是你已 push 到自己的远端）
git clone <你的仓库地址> av_hubert_train && cd av_hubert_train

# ② 在云机器上下载数据集（别从本机传，上传整个数据集又慢又贵）
#    用 AI Studio 的下载链接直接下到云机器，或先传到网盘中转
ls -d main/*/    # 确认是 train / val / test

# ③ 装环境
conda env create -f environment.yml && conda activate avhubert
bash scripts/setup_env.sh

# ④ 下载预训练权重（云机器上直接拉，1.18 GB，很快）
mkdir -p pretrained && cd pretrained
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter4.pt
cd ..

# ⑤ 数据流水线
export LRS2_ROOT=/root/autodl-tmp/main
export DATALIST=/root/autodl-tmp/lrs2_datalist
export WORK=/root/autodl-tmp/lrs2_data
export FFMPEG=$(which ffmpeg)
export DLIB=/root/autodl-tmp/dlib
export NSHARD=8
bash scripts/fetch_dlib_models.sh "$DLIB"
bash scripts/pipeline_lrs2.sh all

# ⑥ 先冒烟测试，再正式微调
export DATA=$WORK/data
export CKPT=$(pwd)/pretrained/base_vox_iter4.pt
export EXP=$(pwd)/exp
bash scripts/run_finetune.sh smoke
NGPU=1 bash scripts/run_finetune.sh train      # 单卡就写 1

# ⑦ 测试并拿点数
bash scripts/run_finetune.sh test "$EXP/train/checkpoints/checkpoint_best.pt" test
cat "$EXP/decode_test"/wer.*      # 文件名是 wer.<哈希>，不是 wer.test
```

#### 实机环境记录（恒源云 RTX 3090 实例，已实测）

拿到实例后跑了一遍完整探测，结果记录在此，便于对照排查：

| 项目 | 实测值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090，24576 MiB，**compute_cap 8.6** |
| 驱动 / CUDA | 驱动 550.163.01；`/usr/local/cuda-11.7`，`cuda.h` 中 `CUDA_VERSION 11070` |
| Python | 3.8.10（系统）+ conda 3.8.20（新建环境） |
| conda | 23.5.0，路径 `/usr/local/miniconda3`（**不在默认 PATH**，需 `export PATH=/usr/local/miniconda3/bin:$PATH`） |
| **conda solver** | **只有 `classic`，没有 libmamba** —— conda-forge 求解会很慢，注意留足超时 |
| 镜像源 | conda / pip 已预配清华与阿里云源 |
| 系统盘 | `/` overlay 30 GB（**别放数据**） |
| 数据盘 | `/hy-tmp` xfs，**容量需在控制台扩容**（默认 50 GB，可扩到 100 GB） |
| 工具 | `git`、`ffmpeg`(需 apt 装)、`oss`(恒源云定制版，含 `login`)、`rclone`、`unzip`、`7z` |
| **缺失** | `rsync`、`zstd`、`pigz`、`pv` —— 传包时注意用 gzip 而非 zstd |

**已验证的兼容性结论**（最关键的一条）：

```
torch 1.13.1+cu117  ->  cuda_avail True
arch list           ->  ['sm_37','sm_50','sm_60','sm_70','sm_75','sm_80','sm_86']
device capability   ->  (8, 6)   ==  sm_86   ✓ 在列表内
```

即「RTX 3090 可用于本项目」这一判断已在真实实例上得到验证。

#### 产物体积估算（决定 `/hy-tmp` 要多大）

由 48,164 条 clip、平均单个 mp4 143 KB 反推：每条约 2 秒、约 500 kbps
（125 KB / 2 s ≈ 500 kbps，× 48,164 ≈ 6.94 GB）。实测 `main/` 目录 7,269,956,461 字节
≈ **7.27 GB**（= 6.77 GiB，早先写「6.77 GB」是把 GiB 当成了 GB），48,165 个 mp4 + 48,165 个 txt。

| 产物 | 估算体积 |
| --- | --- |
| 原始 `main/`（mp4+txt） | **7.27 GB**（= 6.77 GiB，实测 7,269,956,461 字节） |
| `audio/`（16 kHz mono wav） | ~3.1 GB |
| `landmark/`（68 点 pkl） | **1.5 GB**（实测） |
| `video/`（96×96 ROI mp4） | **816 MB**（实测，比预估小很多） |
| manifest / 词表 | < 20 MB |
| **合计** | **约 12.4 GB**（实测：7.27 + 3.3 + 1.5 + 0.82 GB + manifest；早先预估 16–21 GB 偏大） |

总帧数实测 **2,611,599 帧**（约 260 万）。dlib 关键点检测**单进程实测约 0.45 it/s（2.2 s/clip）**，
串行跑完 48,164 条约需 **30 小时**——这是整条流水线的真正瓶颈，
必须靠分片并行（`pipeline_lrs2.sh` 的 `all` 模式已改为阶段内 8 rank 并行，
见 `docs/预处理并行化方案.md` 执行记录）。
因此 **`/hy-tmp` 建议 ≥ 100 GB**：留出「原始数据 + 全部产物 + 训练 checkpoint」的余量。

#### 数据上传的实测参数（重要，避免走弯路）

把 7.27 GB（6.77 GiB）数据集传到实例，实测了几种方式：

| 方式 | 吞吐 | 结论 |
| --- | --- | --- |
| 单连接 SFTP，**`pipelined=True`** | **1.70 MB/s** | ✅ 最优，约 66 分钟 |
| 单连接 SFTP，`pipelined=False` | 0.42 MB/s | ❌ 慢 4 倍 |
| 2 连接并行 | 1.69 MB/s（合计） | 无收益 |
| 8 连接并行 | 网关直接拒绝 | ❌ **会触发限流，禁止** |

**三个结论**：

1. **`pipelined=True` 是决定性的**（0.42 → 1.70 MB/s，4 倍差距），连接数不是
2. **并行连接无收益**：带宽瓶颈在链路上，2 连接合计与单连接相同
3. **并发连接数过高会被网关拒绝**（`Error reading SSH protocol banner`）。
   实测 8 连接时全部握手失败，且之后连单连接也会被短暂拒绝，
   **不要开 8 条以上连接**；用 `parallel_upload.py --conns 1` 即可

另外注意：**实例下行很快（实测清华镜像 60 MB/s），慢的只有「本机→实例」这条路**。
如果数据已存在于平台侧（个人数据 / 公网），让实例自己去拉会快得多。

#### 选卡的注意事项（这里很容易踩坑，务必看）

**关键概念：CUDA 版本 ≠ 算力（compute capability）。** 两者要分别满足：

- **驱动/CUDA 版本**：驱动够新即可，旧版 CUDA 运行时通常能跑（PyTorch 自带运行时）
- **算力 sm_XX**：`torch` 预编译包里有一份**固定的架构列表**，卡的算力不在列表里，
  kernel 就跑不起来。这个才是真正的坑

本机 `torch 2.11.0+cu128` 的列表长这样（实测）：

```
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

而 `torch 1.13.1+cu117`（本环境用的版本）的列表**最高只到 sm_86**。
对照各类卡的算力：

| 显卡 | 算力 | 能否用 `torch 1.13.1+cu117` |
| --- | --- | --- |
| Tesla V100 | sm_70 | ✅ |
| RTX 2080 Ti | sm_75 | ✅ |
| **A100** | **sm_80** | ✅ **推荐** |
| RTX 3090 | sm_86 | ✅ **推荐（性价比高）** |
| RTX 4090 | sm_89 | ❌ 不在列表内 |
| H100 | sm_90 | ❌ 不在列表内 |
| 本机 5060 Laptop | sm_120 | ❌ 不在列表内 |

> ⚠️ 我最初误以为「4090(sm_89) / H100(sm_90) 可用」，那是错的——
> 它们**同样超出 sm_86**。请选 **A100 / 3090 / 2080 Ti / V100** 这类卡。
> 租机器前也可直接问平台客服「是否支持 CUDA 11.7 + torch 1.13」。

**租卡前先验证（30 秒，强烈建议）：**

```bash
# 在租好的机器上、装完环境后执行，确认本卡在该 torch 的架构列表内
python -c "import torch; print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_capability(0))"
```

若输出的算力（如 `(8, 6)`）对应的 `sm_86` 不在列表里，说明这张卡用不了，
**立刻换机器**，不要继续往下跑。

本项目已把上面这套判断写成脚本，在云机器上直接跑：

```bash
python scripts/check_gpu_compat.py
```

它会打印 GPU / 驱动 / torch / 架构列表，给出「能用 / 不能用」的明确结论，
并在不可用时列出该换哪些卡。**建议租好机器后第一件事就跑它**，再决定要不要下数据。

其它注意事项：

- **显存 ≥ 16 GB 最稳**，24 GB（如 3090）更宽裕；8 GB 跑不动 `base` 微调
- 选**国内**平台和机房，否则下载 LRS2 会很慢
- 优先选**能免费/低价换机型**的平台，万一算力不匹配可以马上换

#### 平台候选（均已逐一验证可访问）

> 下面是**候选名单**，不是背书。价格与卡型库存变化很快，
> 请以平台当日「算力市场」页面为准——本节刻意不写具体单价，避免给你过时数字。

| 平台 | 地址 | 特点 / 适合场景 |
| --- | --- | --- |
| **AutoDL** | <https://www.autodl.com> | 国内最主流的按量租用平台，卡型最全（含 3090/2080Ti/A100/V100，均在可用算力范围内）。**首选**；学术邮箱有时有优惠 |
| **共绩算力** | <https://www.gongjiyun.com> | 主打闲时算力，价格可能更低，适合能接受排队/规格波动的场景 |
| **优云智算** | <https://www.compshare.cn> | 按量 GPU 容器，界面友好，适合快速起机跑通流程 |
| **蓝耘** | <https://www.lanyun.net> | 老牌智算平台，机型偏企业级，适合需要稳定长时任务的场景 |
| **恒源云** | <https://www.gpushare.com> | 老牌按量平台，与 AutoDL 定位接近，可作备选（对比同卡价格） |
| **极智算** | <https://www.jygpu.com> | 算力租赁聚合/比价类站点，可用于横向比价后再下单 |

**怎么选（针对本任务）**

1. **先按卡型筛**：只要能提供 **3090（sm_86）/ A100（sm_80）/ 2080 Ti（sm_75）/ V100（sm_70）**
   的平台都行。你的选择面其实很宽——**恰好本项目可用的这几张卡都是各平台的常见机型**，
   不存在"被卡型限制选不到平台"的问题。
2. **再按价格与库存比**：同一张卡不同平台差价比时长更重要；
   3090 通常是最便宜的 24GB 级选择。
3. **优先选支持"无卡模式"或按量计费**的平台：数据预处理阶段（纯 CPU）不需要占卡，
   用无卡模式能省不少钱。
4. **避开 4090/4090D/H100/50 系**：它们算力超出 `torch 1.13.1` 支持范围，
   本环境用不了（除非你愿意自行升级 torch 并承担兼容风险）。

#### 成本与数据摆放

- 把数据集、`WORK`、`EXP` 都放在**大容量数据盘**（如 AutoDL 的 `/root/autodl-tmp`），
  系统盘通常只有几十 GB，装不下大体积的中间产物
- 预处理产物（`audio/`、`video/`、`landmark/`）体积可能接近原始视频，
  预留空间要打得宽一些
- **数据预处理阶段用无卡模式**（纯 CPU 任务），只在微调和测试时才占卡
- 先跑 `bash scripts/run_finetune.sh smoke` 再开正式训练：20 步就能暴露
  数据格式、权重加载、显存不足等问题，避免烧几小时机时才失败

#### 关于直接测试官方权重（不微调）

官方也提供了**已在 LRS3 上微调过**的权重（如 `base_vox_433h.pt`），
可以跳过微调直接解码 LRS2。但 LRS3 → LRS2 存在域不匹配，
WER 大概率是几十个百分点，**不是一个能交差的点数**。建议仅作为链路验证手段。

---

## 5. 数据准备

### 5.0 数据集下载

**只用 main，不要下完整版 LRS2**（题目 `Dataset.md` 明确说明：原始 LRS2 太大，
本考核使用其中的 **main** 部分）：

- 下载地址：<https://aistudio.baidu.com/datasetdetail/132643/0>（百度 AI Studio，需登录）
- 该数据集就是 `main`，包含 `train` / `val` / `test` 三个部分，与 `lrs2_datalist/`
  里三个列表一一对应。

**为什么不需要 pretrain**：官方 433h 预训练训练集依赖完整 LRS2 里的 `pretrain` 部分
（未切分的长时间原始视频，体积远大于 main），而本考核提供的 main 里没有它。
这不是缺陷——`Dataset.md` 已提示「由于数据集缺少 pretrain 部分，部分代码需要相应调整」，
我们的路线就是**用官方预训练权重 + 在 main 上微调 + 在 test 上测试**，
从而完全绕开 pretrain。

**下载后请先核对目录结构**（`lrs2_prepare.py` 依赖这个约定）。

> ⚠️ **实测结论：本考核的数据集是「扁平」结构**，即
> `main/<video_id>/<clip_id>.mp4`，**没有 `train/` `val/` `test/` 子目录**。
> 所以下面这份带划分目录的示意**不适用**于本考核数据，仅供另一种打包方式参考。
> 扁平布局直接用 `--layout flat`（默认值），详见 §5.1。

若是带划分子目录的打包方式，结构应形如：

```
main/
├── train/<video_id>/<clip_id>.mp4      # 每个 clip 一个 mp4
├── train/<video_id>/<clip_id>.txt      # 同名的逐句标注
├── val/<video_id>/<clip_id>.{mp4,txt}
└── test/<video_id>/<clip_id>.{mp4,txt}
```

两种布局通用的判别与校验命令：

```bash
# 1. 看顶层到底是划分目录还是 video_id 目录
ls main/ | head
#   输出 train/val/test        -> 用 --layout split
#   输出一串很长数字(video_id) -> 用 --layout flat（本考核即为此）

# 2. 抽查一个 clip：mp4 与同名 txt 是否都在
ls main/5535415699068794046/00001.*

# 3. 抽查 txt 内容（本数据集格式为 Text: + Conf:）
head -3 main/5535415699068794046/00001.txt
```

`lrs2_prepare.py` 会逐条校验 45,839 + 1,082 + 1,243 条的 mp4 与标注是否齐全，
**缺任何一条都会报错退出**（除非加 `--allow-missing`），因此不需要你自己写检查脚本。
结构不一致时它会打印缺失示例，按提示调整 `--lrs2-root` 层级即可。

### 5.1 与 LRS3 的差异（务必先读）

官方只提供 LRS3 / VoxCeleb2 的预处理脚本，直接套用会踩以下坑：

| 差异 | 说明 | 处理 |
| --- | --- | --- |
| LRS2 无需切句 | main 里每个 clip 已是切好的短句（≤ 约 15 s）并自带标注 | 跳过官方 LRS3 的 step 1/2（按时间戳切长视频） |
| **没有 pretrain 划分** | 题目 `Dataset.md` 已提示；官方 433h 训练集依赖 pretrain | 只用 train/val/test，不再区分 30h / 433h |
| **数据集是扁平结构，没有 `train/val/test` 目录** | 实测从 AI Studio 下载的 main 是 `main/<video_id>/<clip_id>.mp4`，共 **3783 个 video_id** 目录，**没有划分子目录** | `lrs2_prepare.py --layout flat`（默认）。划分归属由 datalist 决定，另产出 `split.list` 供 manifest 使用 |
| **划分目录名是 `val` 而非 `valid`**（仅 split 布局） | 官方 `detect_landmark.py` / `align_mouth.py` 内部按 `{root}/{fid}.mp4` 拼路径 | `file.list` 的 fid **必须**能与实际目录逐字拼接；fairseq 需要的 `valid.tsv` 由 `lrs2_manifest.py` 改名产出 |

> **实测记录（本考核数据集）**：
> - 布局：`main/<video_id>/<clip_id>.mp4`，两层扁平，**无划分子目录**
> - 规模：48,165 个 mp4 + 48,165 个 txt（合计 7.27 GB）；三个列表合计 48,164 条，
>   多出的 1 条（`6375923619026758902/00013`）未被任何列表引用，故产物为 48,164 条
> - 与 datalist 一致性：`train+val+test = 45,839 + 1,082 + 1,243 = 48,164` 条，**逐条核对全部命中，0 缺失**
> - 标注格式：`.txt` 内容为 `Text:  <英文转写>` + `Conf:  <置信度>`，解析后得到干净转写
> - 因此 `--layout` 默认值设为 `flat`；若你的数据带划分子目录请显式加 `--layout split`

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
export LRS2_ROOT=/path/to/lrs2/main          # 指向 main 目录本身（扁平结构，无划分子目录）
export DATALIST=/path/to/lrs2_datalist
export WORK=/path/to/lrs2_data               # 输出都放这里（云上建议放 /hy-tmp）
export FFMPEG=$(which ffmpeg)
export DLIB=/path/to/dlib
export NSHARD=8                              # 并行分片数：**看 cgroup 配额，别信 $(nproc)**
                                             # 本实例 nproc=64，但 cgroup 配额只有 8 核
                                             # （cpu.cfs_quota_us/cpu.cfs_period_us = 800000/100000）
                                             # 设成 64 只会增加上下文切换，不会更快
export LAYOUT=flat                           # 本考核数据集为 flat；带划分子目录时改 split

bash scripts/fetch_dlib_models.sh "$DLIB"    # 下载 dlib 模型 + 20words_mean_face.npy
bash scripts/pipeline_lrs2.sh all
```

流水线共 6 步：

1. `lrs2_prepare.py` —— 解析三个列表 → `file.list` / `label.list` / `split.list`
   （严格校验缺标注 / 缺 mp4；`split.list` 记录每条的划分归属，flat 布局必需）
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

# 2) 正式微调（单卡自动把 world_size 设为 1；多卡时 NGPU=卡数）
NGPU=1 bash scripts/run_finetune.sh train

# 2') 或者：快速微调（单卡 3090 赶时间用这个；省时改造与实测见
#     docs/训练加速与阻塞修复记录.md）
bash scripts/run_finetune.sh fast

# 3) 在 test 集上做纯视觉唇读解码，输出 WER
bash scripts/run_finetune.sh test "$EXP/train/checkpoints/checkpoint_best.pt" test
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

结果写入 `{results_path}/wer.<哈希>`，内容形如 `WER: 12.34`。**这就是题目要汇报的「点数」。**
（注意：文件名不是 `wer.test`，而是 `wer.` 加上 generation 配置的哈希，
见 `infer_s2s.py:247-259`；逐句假设在同目录的 `hypo-<哈希>.json`。
`scripts/run_finetune.sh test` 已用通配符自动找到并打印。）

---

## 7. 评测口径（重要，建议两个都报）

`test.txt` 的 1,243 条自带 `NF` / `MV` 标记（来自 LRS2 的 pretrain 部分）。
`lrs2_manifest.py` 已额外生成 `test_mv.{tsv,wrd}`，可直接：

```bash
bash scripts/run_finetune.sh test "$EXP/train/checkpoints/checkpoint_best.pt" test_mv
```

建议同时汇报：

- **全量 1,243 条**（主口径）
- **仅 `MV`（mouth-visible）子集**（LRS2 论文常用的干净口径）

并写清所用预训练权重档次、微调步数、解码 beam size。

---

## 8. 已知风险与后续事项

1. **算力**：LRS2 main 训练集 45,839 条 ≈ **29 小时**（不是「100+ 小时」）。实测口径：
   **单卡 RTX 3090，8,000 步微调只用 44.3 分钟**（0.33 秒/步）；按官方 base 配方
   （45,000 步 + 后半程解冻编码器）估约 4–6 小时。
   算力有限时可降低 `optimization.max_update`（如 10000），或先用 `--max-train` 跑子集。
2. **本机无法运行**（已实测，详见 [4.1](#41-为什么不在本机跑推荐路线租云-gpu)）：
   Python 3.12 导入 fairseq 直接报错；本机 sm_120 显卡不被 `torch 1.13.1+cu117` 支持；
   8 GB 显存不够微调。**推荐租云 GPU**，本机只做代码维护与版本管理。
3. **选卡必须核对算力 sm_XX**：`torch 1.13.1+cu117` 的架构列表最高到 **sm_86**，
   所以 4090(sm_89)、H100(sm_90)、50 系(sm_120) **都用不了**。
   请选 A100(sm_80) / 3090(sm_86) / 2080 Ti(sm_75) / V100(sm_70)。
   详见 [4.1 选卡的注意事项](#41-为什么不在本机跑推荐路线租云-gpu)。
4. **`test.txt` 的标签列**不是转写文本，只是 `NF` / `MV` 标记，转写需从数据集的 `.txt` 取得。
5. **`val` vs `valid` 命名**：自行改脚本时务必保持「fid 前缀 = 真实目录名」，
   否则 `detect_landmark.py` / `align_mouth.py` 会报 `File does not exist`。
6. **帧数文件顺序**：`nframes.audio` / `nframes.video` 必须与 `file.list` 逐行对应。
7. 官方 checkpoint 全部基于 **LRS3 / VoxCeleb2**（英文），LRS2 同为英文，可直接微调。

---

## 9. 参考

- 论文：*Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction*，<https://arxiv.org/abs/2201.02184>
- 代码：<https://github.com/facebookresearch/av_hubert>
- 权重：<https://facebookresearch.github.io/av_hubert/>
- LRS2 数据：<https://aistudio.baidu.com/datasetdetail/132643/0>
- 官方预处理说明：`avhubert/preparation/README.md`
