# Cosmos3-Nano-Policy-DROID 数据管线与激活检查点优化报告

日期：2026-07-30

工作分支：`perf/nano-policy-droid-pipeline-ac`

上游基线：`origin/main@5e67049cd94acb667786f1e6dd0dab821cb90c97`

## 1. 结论

[百度文章](https://cloud.baidu.com/article/7783486)提到的三个方向都曾经存在，但截至本报告所用的最新上游版本，状态是“部分修复”，不能简单归类为全部已修复或全部仍存在。

| 问题 | 文章发表前后的代码证据 | 最新上游状态 | 本分支处理 |
| --- | --- | --- | --- |
| DROID 重复读取、字段冗余和 Python 对象膨胀 | 历史实现先把每个数据 Parquet 全列转成 Python dict，再由 DROID 路径按所需列重读；见 `24300b4:cosmos_framework/data/vfm/action/datasets/base_dataset.py:65-83` 和 `24300b4:cosmos_framework/data/vfm/action/datasets/droid_lerobot_dataset.py:122-155` | **主要 OOM 根因已修**：`8ea4318`（2026-07-07）改成 metadata/index eager、数据 shard lazy + LRU；当前入口见 `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py:743-803`。但 LeRobot 数据对象仍会读取其 metadata 声明的全部非视频列，**物理 Parquet 列裁剪尚未实现** | 根据 action space、state 和 viewpoint 生成最小列集，并通过 HF Datasets/PyArrow 将 `columns=` 下推到 Parquet；见 `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py:236-339`、`cosmos_framework/data/generator/action/datasets/droid_lerobot_dataset.py:215-233` |
| `max_samples_per_batch > prefetch_capacity` 和 CPU ColorJitter | 文章所对应旧配置可形成 `128 > 1×4×4=16` 的 8 倍供给缺口 | **供给缺口已修**：`8ae459f`（2026-07-28）后的 canonical TOML 是 32 samples/rank，而 inner DataLoader 为 `16×16×2=512`；但 512 个 samples 的队列上界过大。ColorJitter 仍在 DataLoader worker 的 CPU 路径执行，见基线 `5e67049:cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py:165-206` | Python config 与 TOML 都统一为 32 samples/rank，inner queue 改为 `2×8×2=32`；见 `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py:165-187`、`examples/toml/sft_config/action_policy_droid_nano.toml:49-55`。新增 recipe-gated deferred CUDA augmentation |
| Full AC、SAC 和 layer-wise AC | 最新 DROID TOML 仍选择 Full AC；已有 SAC 只用 `["fmha"]` 匹配 `func.__name__`，见 `5e67049:examples/toml/sft_config/action_policy_droid_nano.toml:34-36` 和 `5e67049:cosmos_framework/model/generator/mot/parallelize_unified_mot.py:227-248` | **未修**：Full AC 仍覆盖所有 transformer block；没有按层选择。旧 SAC 规则无法命中 H20 实际出现的 `flash_attn_3._flash_attn_forward.default`，也没有文章描述的 layer-wise 控制 | DROID recipe 切到 backend-aware per-op SAC。策略保存 Flash3/NATTEN/cuDNN/SDPA/FlexAttention、昂贵 ATen op 和 collectives，廉价 op 重算，并按 TorchTitan 策略每隔一个 mm/linear 重算；见 `cosmos_framework/model/generator/mot/parallelize_unified_mot.py:41-164`、`:351-369` |

需要特别区分：

- 文章实际描述的是 **Layer-wise AC**，即按 Transformer layer 决定是否 checkpoint；它没有说“SAC 会缓存 attention layer output”，也没有要求 CV-CUDA。
- 本分支实现的是 **per-op SAC**：保存昂贵算子的输出、重算便宜算子。它解决当前 SAC 规则无效的问题，但不是文章中的 layer-wise AC。
- 最新上游已经消除了最严重的全量 `.to_pylist()` 启动 OOM；本分支的列裁剪是在这个 lazy 架构上的增量优化。

## 2. 修改内容

### 2.1 Parquet 列裁剪

`DROIDLeRobotDataset` 现在只查询当前模式真正会消费的时序字段：

- joint policy：四个 LeRobot 必需 frame 字段，加 joint/gripper action 和 joint/gripper state，共 8 列；
- Cartesian policy：只保留 Cartesian state/action 及需要的 gripper 字段；
- 视频仍从 MP4 读取，不错误地加入 Parquet projection；
- `val_temp_seg` 额外保留其评分需要的 Cartesian state。

投影 loader 同时支持 episode predicate pushdown，并在缺列时给出明确 schema 错误：`cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py:256-339`。另外修复了一个既有的多 shard `val_temp_seg` 错误：原实现固定从 shard 0 取数据，现在按每条 record 的 `ds_idx` 切换并缓存 shard，见 `cosmos_framework/data/generator/action/datasets/droid_lerobot_dataset.py:299-363`。

### 2.2 有界 prefetch 与 GPU ColorJitter

新路径仍在 worker 中解码视频并采样一次 crop/ColorJitter 随机参数，但不再在 CPU 上展开完整的 crop、resize、ColorJitter、三视角拼接和 pad。worker 只打包紧凑的 `[9,T,H,W] uint8` 视频和小型 metadata：`cosmos_framework/data/generator/action/droid_gpu_augmentation.py:47-123`。

数据到达模型设备后，按原有顺序执行：

`crop → bilinear resize → ColorJitter → 外部视角下采样 → 三视角拼接 → uint8 → bicubic resize → pad`

实现见 `cosmos_framework/data/generator/action/droid_gpu_augmentation.py:126-260`。为避免 Hue 转换的临时显存随全部帧增长，默认每 8 帧执行一块；随机顺序和 factors 对所有 camera/frame 仍共享。逻辑分辨率在 worker 阶段只记录、不展开，packer 使用最终逻辑尺寸计 token，见 `cosmos_framework/data/generator/action/transforms.py:404-462` 和 `cosmos_framework/data/generator/joint_dataloader.py:544-567`。模型在既有 normalize 之前消费 deferred augmentation，见 `cosmos_framework/model/generator/omni_mot_model.py:3485-3489`、`:3594-3664`。

没有引入 CV-CUDA 依赖：当前 torchvision v2 CUDA op 已能保持现有 Python transform 的语义，集成风险更小。若后续 profile 证明这一段仍是瓶颈，可以再比较 [CV-CUDA 官方 CUDA 13 wheel](https://cvcuda.github.io/CV-CUDA/installation.html)，但它是第二阶段 backend 选择，不是文章结论的前提。

### 2.3 Backend-aware selective AC

旧 `["fmha"]` 模糊规则改为：

- 使用 Functorch 的 compute-intensive op 表；
- 精确覆盖 Cosmos 当前可能使用的 FlashAttention 2/3、NATTEN、cuDNN fused attention、SDPA/FlexAttention；
- 保存 distributed collectives，避免 backward recompute 重复通信；
- 保存奇数次 mm/linear、重算偶数次 mm/linear；
- 用户 regex 仅作为 namespace-qualified 自定义 backend 的附加规则。

实现和 recipe 分别见 `cosmos_framework/model/generator/mot/parallelize_unified_mot.py:41-164`、`examples/toml/sft_config/action_policy_droid_nano.toml:34-38`。策略参考 [TorchTitan activation checkpoint policy](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/activation_checkpoint.py)，checkpoint API 语义参考 [PyTorch checkpoint 文档](https://docs.pytorch.org/docs/2.13/checkpoint.html)。

## 3. 测试和收益

### 3.1 Parquet：真实 DROID shard

数据：313,318 rows、77.13 MiB 的一个真实 DROID shard；对比 LeRobot 全 17 列和 joint-policy 投影 8 列。

| 指标 | 全 17 列 | 投影 8 列 | 改善 |
| --- | ---: | ---: | ---: |
| dataset init（median） | 0.343 s | 0.169 s | 50.7% |
| materialize 10k rows（median） | 1.643 s | 0.806 s | 50.9% |
| Arrow table | 86.055 MiB | 27.490 MiB | 68.1% |
| HF cache | 86.420 MiB | 27.655 MiB | 68.0% |
| 相同 import baseline 后增量 RSS | 452.2 MiB | 166.6 MiB | 63.2% |

同一真实 episode 0 的 full/projected required values 完全相同，三个视频张量均为 `(33,3,360,640)`。投影数据继续通过 augmentation、normalization 后得到 CUDA tensor `(1,3,33,544,736)`。

作为历史根因对照，旧 `.to_pylist()` 路径只处理这一个 shard 就需要 11.0 s、峰值 RSS 1727.5 MiB；即使手工裁成 8 列后仍需 3.16 s、642.2 MiB。因此最新上游的 lazy 修复贡献最大，本分支不能被表述成再次获得文章所称的 37 分钟到 25 秒全部收益。

### 3.2 Augmentation：真实 DROID episode 0

环境：H20、CUDA 13.0、PyTorch 2.10，固定 seed `20260730`；CPU 路径限制为 1 thread。数据为三视角 33 帧、640×360，最终 480p tier。

| 路径 | median |
| --- | ---: |
| 原 CPU v2 完整 augmentation + compose + pad | 3948.73 ms/sample |
| deferred worker pack + RNG | 309.39 ms/sample |
| pinned H2D（65.26 MiB） | 1.249 ms/sample |
| GPU apply，chunk=8 | 14.389 ms/sample |
| GPU model-side wall（含 metadata） | 15.968 ms/sample |
| deferred 总路径估算 | 325.4 ms/sample |

augmentation 路径是约 **12.1×** 加速。最终 uint8 对齐结果为 MAE `2.086e-5/255`、最大像素差 1、PSNR 94.94 dB、99.99792% 像素完全相同；差异来自 CPU/GPU interpolation 的舍入，训练语义保持一致但不是逐 bit 相同。

GPU chunk 对比：

| frame chunk | kernel median | peak allocated |
| ---: | ---: | ---: |
| 8 | 14.39 ms | 512.33 MiB |
| 16 | 13.98 ms | 576.21 MiB |
| 33 | 13.24 ms | 984.57 MiB |

因此默认推荐 chunk 8。旧 CPU 路径入队后的 `(3,33,544,736) uint8` 视频约 37.80 MiB/sample，512-sample queue 的视频 tensor 理论上界约 18.90 GiB/rank；新 deferred representation 为 65.26 MiB/sample，32-sample queue 约 2.04 GiB/rank。两条路径的入队表示不同，这里只比较各自的容量上界，不等同于进程稳定 RSS。

### 3.3 SAC：H20 短序列 production-width microbenchmark

配置：BF16、`torch.compile(dynamic=True)`、两层 production-width Qwen3-VL-8B MoT block（hidden 4096 / intermediate 12288 / 32Q / 8KV / head 128），2048 packed tokens，4 次 warmup + 10 次测量，forward + FP32 MSE + backward。

| AC | median step | mean step | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: |
| Full | 45.826 ms | 46.096 ms | 4.71137 GiB | 5.07617 GiB |
| Selective | 41.660 ms | 41.710 ms | 4.71137 GiB | 5.05078 GiB |

在这个短序列 case 中，Selective AC 的 step latency 降低 **9.09%**，等价吞吐提升 **10.00%**；allocated 显存持平、reserved 降低约 0.5%。每个 phase 观察到 28 个 mm/linear，其中 14 个保存、14 个重算；4 个真实 Flash3 op 全部保存。

这只是验证 SAC policy 有效且能命中真实 backend 的 microbenchmark，不能外推成 36 层、约 45k tokens、8 卡正式训练的最终收益。是否采用仍应以目标 batch/sequence length 下的稳态 iter 和各卡峰值为准。策略单测覆盖 PyTorch 2.10/2.13 两种 checkpoint callback 语义，但本次真实 GPU 执行环境只有 PyTorch 2.10；PyTorch 2.13 仍属于待补的实际运行矩阵。

### 3.4 4-GPU 真实训练 smoke

使用 4 张 H20、真实 DROID 20 episodes / 10,582 frames，FSDP shard=4、replicate=1、Selective AC、deferred GPU augmentation；为把测试限定为功能验证，模型和 tokenizer compile 均关闭，`max_iter=1`、每 rank 1 sample。

- checkpoint 加载：20.33 s；
- iteration 1：38.68 s（包含首轮初始化，不能视为稳态吞吐）；
- rank 0–3 loss：23.9595、26.3205、27.8205、28.5084；
- 四个 rank 均完成 forward/backward/optimizer step，gradient finite，trainer 打印 `Done with training`。

训练循环随后按固定逻辑自动写最终 checkpoint，见 `cosmos_framework/trainer/__init__.py:324-332`。临时 `/tmp` 空间不足导致最终 DCP 保存失败，因此 torchrun 退出码非零；失败发生在训练 iteration 完成之后。约 53 GiB 的不完整临时 checkpoint 已精确删除，没有保留或覆盖用户 checkpoint。这个 smoke 证明 4-GPU 数据、augmentation、SAC、FSDP 组合可以完成一个训练 step，但不宣称 checkpoint-save 流程成功。

## 4. 验证覆盖

- 目标单测：32 passed，1 个 manual CUDA test deselected；
- H20 真实 Flash3 manual test：1 passed；
- canonical TOML schema/dry-run：passed；
- 真实 DROID projection → GPU augmentation → normalization smoke：passed；
- 4-GPU 真实训练 iteration：passed，最终临时 checkpoint save 因 `/tmp` 空间失败并已清理；
- regression 覆盖：Parquet predicate/columns/schema、joint/Cartesian 列集、多 shard `val_temp_seg`、CPU/GPU augmentation 数值、logical token shape、model-side metadata 消费、SAC op policy/API compatibility；测试入口见 `cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot_test.py:88-254`、`cosmos_framework/data/generator/action/droid_gpu_augmentation_test.py:54-162`、`cosmos_framework/model/generator/mot/parallelize_unified_mot_test.py:50-206`。

## 5. 推荐配置与后续正式 benchmark

当前推荐值已经落到 DROID recipe：

```toml
[dataloader_train]
max_samples_per_batch = 32

[model.activation_checkpointing]
mode = "selective"
save_ops_regex = []
```

inner DataLoader 为 `batch_size=2, num_workers=8, prefetch_factor=2`，GPU augmentation chunk 为 8。正式 8 卡 benchmark 应至少运行编译完成后的 20 个 warmup iter 和 100 个测量 iter，同时记录：

1. Data wait p50/p95 和 samples/s/GPU；
2. 稳态 iter p50/p95；
3. 每张 GPU 的 peak allocated/reserved；
4. 每个 worker/rank 的 host RSS 和 pinned memory；
5. Full AC 与 Selective AC 的 loss/gradient 数值一致性。

如果 data wait 出现周期性尖峰，先把 `prefetch_factor` 从 2 调到 3（capacity 48），而不是恢复 512-sample queue。若长序列下 Selective AC 显存超预算，再比较 layer-wise AC；那应作为独立设计，不能与本次 per-op SAC 的结果混为一谈。
