.PHONY: config-check create-run sanity-sft sanity-dpo sanity-grpo sanity-rl sanity-all list-runs

config-check:
	python -m polaris.config --base configs/base.yaml --override configs/qwen3_0_6b/sft_math.yaml --print

create-run:
	python scripts/create_run.py --base configs/base.yaml --override configs/qwen3_0_6b/sft_math.yaml

sanity-sft:
	python scripts/sanity/sanity_sft.py --base configs/base.yaml --override configs/qwen3_0_6b/sft_math.yaml

sanity-dpo:
	python scripts/sanity/sanity_dpo.py --base configs/base.yaml --override configs/qwen3_0_6b/dpo_math.yaml

sanity-grpo:
	python scripts/sanity/sanity_grpo.py --base configs/base.yaml --override configs/qwen3_0_6b/grpo_math.yaml

sanity-rl:
	python scripts/sanity/sanity_ppo.py --base configs/base.yaml --override configs/qwen3_0_6b/grpo_math.yaml

sanity-all: sanity-sft sanity-dpo sanity-grpo sanity-rl

list-runs:
	python -m polaris.registry --runs-dir runs --list
