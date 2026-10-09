#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
make_full_config.py -- 把「只写关键项」的 AV-HuBERT 配置，补全成自包含配置

为什么需要这个脚本
------------------
fairseq 的 `hydra_train.py` 按下面的顺序工作（见 fairseq_cli/hydra_train.py:25-29）：

    @hydra.main(config_path=..., config_name="config")     # 主配置
    def hydra_main(cfg):
        add_defaults(cfg)                                  # 只补 Any 类型的分组
        if cfg.common.reset_logging: ...                   # ← 立刻读 common 的字段

而 `add_defaults`（fairseq/dataclass/initialize.py:41-61）**只**对
`FairseqConfig` 里类型为 `Any` 的分组（task / model / criterion / optimizer /
lr_scheduler / bpe / tokenizer / scoring / generation）做 dataclass 默认值合并；
`common` / `checkpoint` / `dataset` / `distributed_training` 这几个分组是强类型字段，
它们的默认值**只来自主配置**。

平时主配置是 fairseq 自带的 `fairseq/config/config.yaml`，Hydra 会用
`ConfigStore` 里注册的 `FairseqConfig` 结构化配置把这几段补全，所以一切正常。
但 AV-HuBERT 的用法是 `--config-name finetune/xxx`，**主配置被换成了那个 yaml**，
于是 `common` 等分组里只剩 yaml 中写到的键，读到没写的键就报：

    omegaconf.errors.ConfigAttributeError: Missing key reset_logging

甲、乙两种修法里，这里选了「生成自包含配置」：
  * 甲、把 hydra-core 降到 fairseq 要求的 <1.1 —— 动环境，风险大且未必对症；
  * 乙、让配置自己补全 common/checkpoint/dataset/distributed_training 的**全部**字段
       —— 也就是本脚本做的事，不依赖 hydra 版本。

用法
----
    python scripts/make_full_config.py \
        --src avhubert/conf/finetune/base_lrs2_video_only_fast.yaml \
        --out avhubert/conf/finetune/_full_base_lrs2_video_only_fast.yaml

生成的文件带 `# @package _global_`：本文件位于 conf/finetune/ 下，
Hydra ≥1.1 里若用 `_group_`，所有键会被挂到 `finetune.*` 而非顶层。
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="人工维护的配置（只写关键项）")
    ap.add_argument("--out", required=True, help="输出的自包含配置")
    args = ap.parse_args()

    try:
        from omegaconf import OmegaConf
        from fairseq.dataclass.configs import FairseqConfig
    except Exception as e:                                    # pragma: no cover
        print("导入 fairseq/omegaconf 失败（需要在 avhubert conda 环境里跑）：%s" % e,
              file=sys.stderr)
        return 1

    user = OmegaConf.load(args.src)
    full = OmegaConf.structured(FairseqConfig)                # 完整 schema + 全部默认值
    OmegaConf.set_struct(full, False)

    # 人工配置覆盖到完整 schema 上。注意 task/model/optimizer 等在
    # FairseqConfig 里默认是 None（Any 类型），merge 后变成我们写的 dict，
    # 之后的 add_defaults 会再把这些分组的 dataclass 默认值补齐。
    merged = OmegaConf.merge(full, user)

    # hydra 段（run.dir 等）不在 FairseqConfig 里，单独保留
    if "hydra" in user:
        merged.hydra = user.hydra

    txt = OmegaConf.to_yaml(merged, resolve=False)
    header = (
        "# @package _global_\n"
        "# ===== 本文件由 scripts/make_full_config.py 自动生成，请勿手改 =====\n"
        "# 源文件: %s\n"
        "# 作用: 把 common/checkpoint/dataset/distributed_training 补成 fairseq 要求的\n"
        "#       完整字段集。原因见 make_full_config.py 的文档字符串。\n"
        % os.path.basename(args.src)
    )
    with open(args.out, "w") as fo:
        fo.write(header)
        fo.write(txt)

    n_common = len(merged.get("common", {}))
    n_ds = len(merged.get("dataset", {}))
    print("已生成 %s（%d 行；common %d 字段，dataset %d 字段）"
          % (args.out, len(txt.splitlines()) + len(header.splitlines()), n_common, n_ds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
