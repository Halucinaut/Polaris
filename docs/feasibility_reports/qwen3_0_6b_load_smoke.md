Qwen3 0.6B Load Smoke Test

模型权重加载：scripts/smoke/load_model.py
模型权重文件：models/qwen3_0_6b/

本地机器可加载 Qwen3-0.6B MLX bf16 模型并生成简短响应。脚本还提供了无效模型路径的可读性诊断信息。此烟雾测试仅验证模型加载和简短生成功能，不涉及基准准确率、批量推理吞吐量或 SFT 训练可行性。