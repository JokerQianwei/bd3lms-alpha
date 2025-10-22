import itertools, os
from typing import Optional
import datasets, torch
import utils
LOGGER = utils.get_logger(__name__)


def _group_texts(examples, block_size: int, bos: int, eos: int, insert_special_tokens: bool = True):
  """拼接后按块切分，必要时注入[BOS]/[EOS]。"""
  concatenated = list(itertools.chain(*examples["input_ids"]))
  new_block = block_size - 2 if insert_special_tokens else block_size
  total = (len(concatenated) // new_block) * new_block
  out = {"input_ids": [], "attention_mask": []}
  for i in range(0, total, new_block):
    seq = ([bos] + concatenated[i:i+new_block] + [eos]) if insert_special_tokens else concatenated[i:i+new_block]
    out["input_ids"].append(seq); out["attention_mask"].append(torch.ones(block_size))
  return out


def get_dataset_smiles(tokenizer, wrap: bool, mode: str, cache_dir: str, block_size: int,
                       num_proc: int, streaming: bool, insert_eos: bool,
                       insert_special_tokens: bool, raw_data_path: Optional[str] = None):
  """SMILES 专用数据管线：分词/分块/缓存（与原实现等价）。"""

  # 生成缓存文件名
  eos_tag = "_specialFalse" if not insert_special_tokens else ("_eosFalse" if not insert_eos else "")
  style_tag = "_bosEOS" if (not wrap and insert_special_tokens) else ""
  mode_tag = "wrapped" if wrap else "unwrapped"
  _path = os.path.join(cache_dir, f"smiles_{mode}_bs{block_size}_{mode_tag}{eos_tag}{style_tag}.dat")

  if utils.fsspec_exists(_path):
    LOGGER.info(f"Loading SMILES data from: {_path}")
    return datasets.load_from_disk(_path).with_format("torch")

  LOGGER.info(f"Generating SMILES data at: {_path}")
  data = datasets.load_from_disk(raw_data_path or cache_dir)[mode]
  original_count = len(data)

  EOS, BOS = tokenizer.vocab["[EOS]"], tokenizer.vocab["[BOS]"]
  
  def preprocess_and_tokenize(example):
    if "input" in example: text = example["input"]
    elif "text" in example: text = example["text"]

    tokenizer.padding_side = "right"; tokenizer.truncation_side = "right"
    if wrap:
      tokens = tokenizer(text, add_special_tokens=False)
      if insert_eos:
        tokens = {"input_ids": [t + [EOS] for t in tokens["input_ids"]]}
      return tokens
    else:
      # 非 wrap：先无截断分词（不加任何特殊符），返回中间列供后续过滤与构造
      tmp = tokenizer(text, add_special_tokens=False, padding=False, truncation=False, return_attention_mask=False,)
      ids_list = tmp["input_ids"]
      lens = [len(ids) for ids in ids_list]
      return {"_ids": ids_list, "_len": lens}

  tokenized = data.map(preprocess_and_tokenize, batched=True) if streaming else data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)

  # 非 wrap：基于长度过滤 + 构造定长样本
  if not wrap:
    max_len = block_size - 2 if insert_special_tokens else block_size

    def _keep_fn(batch):
      return [l <= max_len for l in batch["_len"]]

    tokenized = tokenized.filter(_keep_fn, batched=True)

    def _build_features(batch):
      pad_id = tokenizer.vocab.get("[PAD]")
      input_ids, attn_masks = [], []
      for ids in batch["_ids"]:
        if insert_special_tokens:
          seq = [BOS] + ids + [EOS]
        else:
          seq = ids
        pad_len = block_size - len(seq)
        input_ids.append(seq + [pad_id] * pad_len)
        attn_masks.append([1] * (block_size - pad_len) + [0] * pad_len)
      return {"input_ids": input_ids, "attention_mask": attn_masks}

    tokenized = tokenized.map(
      _build_features,
      batched=True,
      num_proc=None if streaming else num_proc,
      load_from_cache_file=not streaming,
    )

    # 移除中间与原始列
    cols = [c for c in tokenized.column_names if c not in {"input_ids", "attention_mask"}]
    if cols:
      tokenized = tokenized.remove_columns(cols)

    # 统计丢弃数量
    kept_count = len(tokenized)
    dropped = max(0, original_count - kept_count)
    ratio = (dropped / original_count * 100.0) if original_count > 0 else 0.0
    LOGGER.info(f"SMILES 非 wrap 模式：丢弃超长样本 {dropped}/{original_count} ({ratio:.2f}%).")

    if not streaming:
      tokenized.save_to_disk(_path)
    return tokenized.with_format("torch")

  group_texts = lambda ex: _group_texts(ex, block_size, BOS, EOS, insert_special_tokens)

  chunked = tokenized.map(group_texts, batched=True) if streaming else tokenized.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
  (None if streaming else chunked.save_to_disk(_path))
  return chunked.with_format("torch")
