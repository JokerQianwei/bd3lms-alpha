# Repository Guidelines

## 项目结构与模块组织
- `main.py`：训练/评测入口（Hydra 配置驱动）。
- `configs/`：配置树（data/model/algo/noise/lr/strategy/callbacks）。
- `models/`：模型实现（DiT、AR、HF 适配）。
- `scripts/`：常用脚本（`train/`、`ppl/`、`zs_ppl/`、`gen_ppl/`、`var_len/`）。
- 运行产物目录：`outputs/`、`logs/`、`sample_logs/`、`watch_folder/`。

## 构建、测试与开发命令
- 环境：`conda create -n bd3lm python=3.9 && conda activate bd3lm && pip install -r requirements.txt`
- 初始化目录：`mkdir outputs watch_folder logs sample_logs`
- 冒烟运行（单卡/快速）：
  `python -u main.py model=tiny data=ptb trainer.devices=1 loader.batch_size=1 loader.eval_batch_size=1 trainer.limit_train_batches=0.1 trainer.limit_val_batches=0.1 wandb=null`
- 训练：`sbatch scripts/train/train_owt_bd3lm.sh`
- PPL 评测：`python -u main.py mode=ppl_eval model=small data=openwebtext-split wandb=null`
- 采样评测：`python -u main.py mode=sample_eval block_size=4 model.length=2048 sampling.kv_cache=true sampling.logdir=$PWD/sample_logs/bd3lm_len2048`

## 代码风格与命名约定
- Python 3.9+；PEP8；4 空格缩进；行宽≤100。
- 命名：函数/变量`snake_case`；类`PascalCase`；常量`UPPER_SNAKE`。
- 超参优先放入 `configs/`，避免硬编码；新增字段需默认值与注释。
- 关键路径补充类型注解与 docstring；日志使用 `utils.get_logger`。

## 测试与验证指南
- 本仓库无单元测试套件；以可复现实验为主。
- 最小化验证：使用 `model=tiny data=ptb` 与 `trainer.limit_*_batches` 做 1–2 step 冒烟。
- 指标验证：用 `scripts/ppl/*` 复现 perplexity；`scripts/gen_ppl/*` 与 `scripts/var_len/*` 复现生成质量与变长。

## 提交与 Pull Request
- 建议 Conventional Commits：`feat|fix|refactor|docs|chore(scope): message`。
- PR 必需：变更说明、动机与影响、复现命令、关键日志/截图、关联 issue、兼容性与回滚方案。
- 变更涉及行为/超参时，同步更新 `configs/` 与 `README.md`。

## 安全与配置提示（可选）
- 加载 HF 权重使用 `trust_remote_code=True`，仅信任可靠来源。
- 大规模训练需协调 `trainer.devices/num_nodes` 与 `loader.global_batch_size`，必要时依赖梯度累积避免 OOM。

## Agent 专用说明
- 修改前先查 `configs/` 与 `scripts/`，保持接口/脚本兼容；公共工具放入 `utils.py`。
- 不破坏现有算法分支（`bd3lm`/`ar`/`sedd`/`mdlm`）；新增超参需更新对应 YAML。

