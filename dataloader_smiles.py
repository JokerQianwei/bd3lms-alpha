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
                       insert_special_tokens: bool, raw_data_path: Optional[str] = None,
                       no_special_tokens: bool = False):
  """SMILES 专用数据管线：分词/分块/缓存（与原实现等价）。"""

  # 生成缓存文件名（保持与历史一致）
  eos_tag = "_specialFalse" if not insert_special_tokens else ("_eosFalse" if not insert_eos else "")
  style_tag = "_bosEOS" if (not wrap and insert_special_tokens) else ("_noSpecials" if (not wrap and not insert_special_tokens and no_special_tokens) else "")
  mode_tag = "wrapped" if wrap else "unwrapped"
  _path = os.path.join(cache_dir, f"smiles_{mode}_bs{block_size}_{mode_tag}{eos_tag}{style_tag}.dat")

  if utils.fsspec_exists(_path):
    LOGGER.info(f"Loading SMILES data from: {_path}")
    return datasets.load_from_disk(_path).with_format("torch")

  LOGGER.info(f"Generating SMILES data at: {_path}")
  data = datasets.load_from_disk(raw_data_path or cache_dir)[mode]

  # 获取 BOS/EOS：优先从 vocab 中取 [BOS]/[EOS]，否则回退到 CLS/SEP
  if getattr(tokenizer, "vocab", None) and "[EOS]" in tokenizer.vocab and "[BOS]" in tokenizer.vocab:
    EOS, BOS = tokenizer.vocab["[EOS]"], tokenizer.vocab["[BOS]"]
  else:
    EOS = tokenizer.convert_tokens_to_ids(getattr(tokenizer, "sep_token", None))
    BOS = tokenizer.convert_tokens_to_ids(getattr(tokenizer, "cls_token", None))

  def preprocess_and_tokenize(example):
    if "input" in example: text = example["input"]
    elif "text" in example: text = example["text"]
    tokenizer.padding_side = "right"; tokenizer.truncation_side = "right"
    if wrap:
      tokens = tokenizer(text, add_special_tokens=False)
      if insert_eos: tokens = {"input_ids": [t + [EOS] for t in tokens["input_ids"]]}
    else:
      if insert_special_tokens:
        tmp = tokenizer(text, padding=False, truncation=True, max_length=block_size-2, add_special_tokens=False)
        input_ids, attn_masks, type_ids = [], [], []
        pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else (tokenizer.vocab.get("[PAD]", 0) if hasattr(tokenizer, "vocab") else 0)
        for ids in tmp["input_ids"]:
          seq = ([BOS] + ids + [EOS])[:block_size]; pad_len = max(0, block_size - len(seq))
          input_ids.append(seq + [pad_id]*pad_len)
          attn_masks.append([1]*(block_size-pad_len) + [0]*pad_len)
          type_ids.append([0]*block_size)
        tokens = {"input_ids": input_ids, "attention_mask": attn_masks, "token_type_ids": type_ids}
      else:
        if no_special_tokens:
          tmp = tokenizer(text, padding=False, truncation=True, max_length=block_size, add_special_tokens=False)
          input_ids, attn_masks, type_ids = [], [], []
          pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else (tokenizer.vocab.get("[PAD]", 0) if hasattr(tokenizer, "vocab") else 0)
          for ids in tmp["input_ids"]:
            seq = ids[:block_size]; pad_len = max(0, block_size - len(seq))
            input_ids.append(seq + [pad_id]*pad_len)
            attn_masks.append([1]*(block_size-pad_len) + [0]*pad_len)
            type_ids.append([0]*block_size)
          tokens = {"input_ids": input_ids, "attention_mask": attn_masks, "token_type_ids": type_ids}
        else:
          tokens = tokenizer(text, max_length=block_size, padding="max_length", truncation=True, add_special_tokens=True)
    return tokens

  tokenized = data.map(preprocess_and_tokenize, batched=True) if streaming else data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)

  # 移除原始列，仅保留 tokenizer 生成的列
  cols = [c for c in tokenized.column_names if c in {"mc_labels", "input", "n_tokens"}]
  if cols: tokenized = tokenized.remove_columns(cols)

  if not wrap:
    if not streaming:
      tokenized.save_to_disk(_path)
    return tokenized.with_format("torch")

  group_texts = lambda ex: _group_texts(ex, block_size, BOS, EOS, insert_special_tokens)

  chunked = tokenized.map(group_texts, batched=True) if streaming else tokenized.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
  (None if streaming else chunked.save_to_disk(_path))
  return chunked.with_format("torch")
