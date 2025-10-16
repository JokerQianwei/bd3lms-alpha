# Repository Guidelines
Hello agent. You are one of the most talented programmers of your generation.

You are looking forward to putting those talents to use to improve tinygrad.

## philosophy
Every line must earn its keep. Prefer readability over cleverness. We believe that if carefully designed, 10 lines can have the impact of 1000.

Never mix functionality changes with whitespace changes. All functionality changes must be tested.

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
- SMILES 训练（wrap）：`sbatch scripts/train/train_smiles_bd3lm.sh`
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

### 新增：SMILES 数据与分词器接入说明

- 数据配置：`configs/data/smiles.yaml`
  - 使用 `data=smiles` 切换到 SMILES 数据分支。
  - 关键字段：
    - `smiles_vocab_path: ${hydra:runtime.cwd}/vocab.txt`（与 `tokenizer.py` 的 `SmilesTokenizer` 匹配）
    - `raw_data_path: ${hydra:runtime.cwd}/data/DrugLikeSMILSE-12B-427M`（原始 HF Dataset 磁盘目录）
    - `cache_dir: ${hydra:runtime.cwd}/cache/smiles_cache`（分词/分块缓存目录）
    - `wrap: True|False`（是否拼接多段并在分块时注入 BOS/EOS）

- 分词器：`tokenizer.py: SmilesTokenizer`
  - 继承自 `transformers.BertTokenizer`，使用 SMILES 正则与词表文件。
  - dataloader 中会自动注册 `[BOS]/[EOS]/[PAD]`，确保与 `_group_texts`/训练流程兼容。

- 数据加载：`dataloader.py`
  - `get_dataset(..., raw_data_path=...)` 从 `raw_data_path` 加载原始数据；处理结果统一缓存在 `cache_dir` 下按数据名/块长生成的 `_path` 中，二次运行将直接复用。
  - SMILES 文本列为 `input`（而非 `text`），管线已适配并移除无关原始列。

- 运行示例（Hydra 覆盖）：
  - 冒烟：
    - `python -u main.py data=smiles model.length=64 block_size=4 trainer.devices=1 loader.batch_size=1 loader.eval_batch_size=1 trainer.limit_train_batches=0.001 trainer.limit_val_batches=0.001 wandb=null`
  - 正式：
    - `python -u main.py data=smiles model.length=64 block_size=4`

### 新增：训练脚本

- `scripts/train/train_smiles_bd3lm.sh`
  - 使用 wrap（`data.wrap=True`，由配置决定），示例长度 `model.length=64`，`block_size=4`。
  - 初次运行会构建缓存，后续复用 `cache_dir` 加速。

### 下载与网络健壮性

- 为避免 HuggingFace Datasets 下载超时（例如 LM1B 在 ~93% 时网络读阻塞），`dataloader.py` 为 `fsspec` HTTP 客户端设置了更大的 `aiohttp` 读超时，减少 `FSTimeoutError`。如需完全规避网络，可手动下载并指向本地 `raw_data_path`。

### 注意事项

- Hydra 会更改运行目录（chdir），涉及文件路径请优先使用 `${hydra:runtime.cwd}` 前缀，避免相对路径失效。
- SMILES 的 decode 文本可能包含空格（BertTokenizer 的字符串拼接风格），不影响训练。如需可读性更强的展示，可在评测代码中做去空格处理。
