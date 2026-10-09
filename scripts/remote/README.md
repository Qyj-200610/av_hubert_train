# scripts/remote/ —— 实例侧驱动与归集脚本

这些脚本**不在实例上跑训练**，而是在本机通过 SSH 驱动/监控云实例、并把产物归集回仓库。
它们原先散落在工作区根目录，导致 `docs/` 与 `results/` 里的引用在提交文件夹里找不到
（"证据链断裂"），现在统一收进本目录并随仓库提交。

## 文件

| 文件 | 作用 |
| --- | --- |
| `gh_ssh.py` | **统一建连模块**（上面的脚本都从这里取 `connect` / `run`）。处理 gpushare 网关限流：遇到 `Error reading SSH protocol banner` 时指数退避重试 |
| `collect_results.py` | 把实例上的训练/解码产物（`wer.*`、`hypo-*.json`、日志）归集到仓库的 `results/04_原始产物/` |
| `train_optimize.py` | 训练侧的实验驱动：`timed`（计时）、`synth`（造合成数据集测吞吐）等子命令 |
| `run_parallel_pipeline.py` | 六个子命令：连接 / 上传 / 验证 / 全量 / 监控 / 验收 |
| `test_run_ranks_parallel.py` | 单独验证 `run_ranks_parallel` 的失败传播语义（从 `pipeline_lrs2.sh` 抽出函数 + 桩函数） |
| `pipeline_status.py` | **已废弃**：读的是旧日志路径。保留仅为让文档引用可解析，不要运行 |
| `parallel_upload.py` | **已废弃**：分片续传有已知缺陷（远端文件多出 437 MB 那次）。保留仅为记录实测数据 |

## 凭据

**本目录不含任何口令。** 所有凭据都从环境变量读取：

```bash
export AVH_HOST=<实例地址>
export AVH_PORT=<SSH 端口>
export AVH_USER=<用户名>
export AVH_PW=<口令>
```

也可以写进 `~/.avh_cred`（每行一条 `KEY=VALUE`，权限 600）。
缺凭据时 `gh_ssh.require_cred()` 会给出明确指引，而不是带着空口令去连
（那样只会得到一个迷惑的认证错误）。

## 用法

脚本已改为不依赖"当前目录是工作区根"：`sys.path` 按**脚本自身所在目录**解析，
仓库位置也按脚本路径推导（可用 `AVH_REPO_LOCAL` / `AVH_RESULTS_DIR` 覆盖）。

```bash
python scripts/remote/collect_results.py
python scripts/remote/run_parallel_pipeline.py --help
```

> 早期版本硬编码了 `D:\人工智能组招新题目\...` 这样的本机绝对路径，换机器即失效，已改掉。
