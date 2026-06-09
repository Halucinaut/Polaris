# Polaris

本地优先的 Post-Training 能力训练项目。

## 项目定位

Polaris 是一个在 MacBook Pro M5 Max（128 GB 统一内存）上运行的本地 post-training 实验框架。第一目标是系统掌握 SFT、DPO、GRPO、PPO、OPD 的训练信号、数据构造、实验诊断和硬件边界。

项目采用两条任务线：
- **Math reasoning**（主线）：串通 SFT -> DPO -> GRPO -> PPO 的核心流程
- **Tool-call**（副线）：训练垂类领域微调能力

## 当前阶段：M0（环境骨架搭建）

M0 的目标是让 Polaris 有一个稳定的实验入口和配置协议。此阶段**不下载模型、不跑训练、不接 MLX**。

### M0 范围

| Step | 内容 | 状态 |
|---|---|---|
| Step 1 | 项目元信息与配置协议 | 已完成 |
| Step 2 | run registry 协议 | 已完成 |
| Step 3 | metric / hardware log 协议 | 已完成 |
| Step 4 | experiment card / failure note 模板 | 已完成 |
| Step 5 | sanity 脚本空跑闭环 | 已完成 |
| Step 6 | M0 验收命令集合 | 已完成 |

### M0 不做的事项

- 不下载任何模型权重
- 不执行任何训练（SFT/DPO/GRPO/PPO）
- 不实现 Tool-call 相关代码
- 不接入 MLX 框架

## 目录约定

```
polaris/                    # 项目根
├── configs/                # 实验配置（base + 覆盖）
├── polaris/                # Python 包（配置、注册表、日志协议）
├── scripts/                # 可执行脚本（sanity、训练、评估）
├── runs/                   # 实验输出目录（gitignored）
├── data/                   # 数据目录
├── docs/                   # 实验笔记、方法卡片、模板
└── future/                 # 未来可选方向（云端 OPD 等）
```

## M0 Validation

Polaris M0 validates the local experiment skeleton only. It does not download models, run training, or require MLX/PyTorch.

```bash
make config-check
make sanity-all
make list-runs
```

A valid M0 run should create four completed fake runs: SFT, DPO, GRPO, and RL sanity.

Each run directory should contain:

```
config.yaml
run_meta.yaml
metrics/train_metrics.jsonl
metrics/eval_metrics.jsonl
metrics/hardware_log.jsonl
samples/sample_diff.jsonl
experiment_card.md
metric_report.md
sample_diff.md
failure_note.md
run_report.md
```

### Available Make Targets

| Target | Description |
|---|---|
| `make config-check` | Print merged config (base + override) |
| `make create-run` | Create a single run from config |
| `make sanity-sft` | Run SFT fake sanity |
| `make sanity-dpo` | Run DPO fake sanity |
| `make sanity-grpo` | Run GRPO fake sanity |
| `make sanity-rl` | Run PPO/RL fake sanity |
| `make sanity-all` | Run all four sanity scripts |
| `make list-runs` | List all runs with status |

### Manual Verification

```bash
# Check a specific run
ls runs/<run_id>
cat runs/<run_id>/run_meta.yaml
head runs/<run_id>/metrics/train_metrics.jsonl
head runs/<run_id>/metrics/eval_metrics.jsonl
head runs/<run_id>/metrics/hardware_log.jsonl
head runs/<run_id>/samples/sample_diff.jsonl
```

