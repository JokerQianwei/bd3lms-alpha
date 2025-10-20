import os, sys, pathlib
from typing import Tuple

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import dataloader
from tokenizer import SmilesTokenizer


def _ensure_special_tokens(tok: SmilesTokenizer) -> SmilesTokenizer:
    """Ensure BOS/EOS/PAD are registered on tokenizer for SMILES tests."""
    if tok.bos_token is None and '[BOS]' in getattr(tok, 'vocab', {}):
        tok.add_special_tokens({'bos_token': '[BOS]'})
    if tok.eos_token is None and '[EOS]' in getattr(tok, 'vocab', {}):
        tok.add_special_tokens({'eos_token': '[EOS]'})
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': '[PAD]'})
    return tok


def _bos_eos_ids(tok: SmilesTokenizer) -> Tuple[int, int]:
    if getattr(tok, 'vocab', None) and '[EOS]' in tok.vocab and '[BOS]' in tok.vocab:
        return tok.vocab['[BOS]'], tok.vocab['[EOS]']
    return (
        tok.convert_tokens_to_ids(getattr(tok, 'cls_token', None)),
        tok.convert_tokens_to_ids(getattr(tok, 'sep_token', None)),
    )


def test_smiles_non_wrap_pipeline(tmp_path):
    raw = 'data/DrugLikeSMILSE-debug'
    assert os.path.isdir(raw), 'local HF dataset not found: data/DrugLikeSMILSE-debug'

    tok = _ensure_special_tokens(SmilesTokenizer(vocab_file='vocab.txt'))
    BOS, EOS = _bos_eos_ids(tok)

    cache_dir = tmp_path.as_posix()
    ds = dataloader.get_dataset(
        dataset_name='smiles', tokenizer=tok, wrap=False, mode='train', cache_dir=cache_dir,
        block_size=64, num_proc=1, streaming=False, insert_eos=True, insert_special_tokens=True,
        raw_data_path=raw,
    )

    assert len(ds) > 0
    sample = ds[0]
    assert set(sample.keys()) == {'input_ids', 'attention_mask', 'token_type_ids'}
    assert isinstance(sample['input_ids'], torch.Tensor)
    assert sample['input_ids'].shape[0] == 64
    # 最后一个有效位置应为 EOS
    valid = int(sample['attention_mask'].sum().item())
    assert valid > 2
    assert int(sample['input_ids'][valid - 1].item()) == EOS
    # 首 token 为 BOS
    assert int(sample['input_ids'][0].item()) == BOS


def test_smiles_wrap_pipeline(tmp_path):
    raw = 'data/DrugLikeSMILSE-debug'
    assert os.path.isdir(raw), 'local HF dataset not found: data/DrugLikeSMILSE-debug'

    tok = _ensure_special_tokens(SmilesTokenizer(vocab_file='vocab.txt'))
    BOS, EOS = _bos_eos_ids(tok)

    cache_dir = tmp_path.as_posix()
    ds = dataloader.get_dataset(
        dataset_name='smiles', tokenizer=tok, wrap=True, mode='train', cache_dir=cache_dir,
        block_size=64, num_proc=1, streaming=False, insert_eos=True, insert_special_tokens=True,
        raw_data_path=raw,
    )

    assert len(ds) > 0
    sample = ds[0]
    assert set(sample.keys()) == {'input_ids', 'attention_mask'}  # 无 token_type_ids
    assert sample['input_ids'].shape[0] == 64
    # wrap 分块后首尾应为 BOS/EOS
    assert int(sample['input_ids'][0].item()) == BOS
    assert int(sample['input_ids'][-1].item()) == EOS


def _expected_cache_path(cache_dir: str, wrap: bool, mode: str = 'train', block_size: int = 64,
                         insert_eos: bool = True, insert_special_tokens: bool = True,
                         no_special_tokens: bool = False) -> str:
    eos_tag = "_specialFalse" if not insert_special_tokens else ("_eosFalse" if not insert_eos else "")
    style_tag = (
        "_bosEOS" if (not wrap and insert_special_tokens) else
        ("_noSpecials" if (not wrap and not insert_special_tokens and no_special_tokens) else "")
    )
    mode_tag = "wrapped" if wrap else "unwrapped"
    return os.path.join(cache_dir, f"smiles_{mode}_bs{block_size}_{mode_tag}{eos_tag}{style_tag}.dat")


def test_smiles_caching_outputs(tmp_path):
    raw = 'data/DrugLikeSMILSE-debug'
    assert os.path.isdir(raw), 'local HF dataset not found: data/DrugLikeSMILSE-debug'

    tok = _ensure_special_tokens(SmilesTokenizer(vocab_file='vocab.txt'))
    cache_dir = tmp_path.as_posix()

    # 非 wrap：应保存到磁盘
    _ = dataloader.get_dataset(
        dataset_name='smiles', tokenizer=tok, wrap=False, mode='train', cache_dir=cache_dir,
        block_size=64, num_proc=1, streaming=False, insert_eos=True, insert_special_tokens=True,
        raw_data_path=raw,
    )
    path_unwrapped = _expected_cache_path(cache_dir, wrap=False)
    assert os.path.exists(path_unwrapped), path_unwrapped

    # wrap：也应保存
    _ = dataloader.get_dataset(
        dataset_name='smiles', tokenizer=tok, wrap=True, mode='train', cache_dir=cache_dir,
        block_size=64, num_proc=1, streaming=False, insert_eos=True, insert_special_tokens=True,
        raw_data_path=raw,
    )
    path_wrapped = _expected_cache_path(cache_dir, wrap=True)
    assert os.path.exists(path_wrapped), path_wrapped


def test_smiles_non_wrap_no_special_tokens(tmp_path):
    raw = 'data/DrugLikeSMILSE-debug'
    assert os.path.isdir(raw), 'local HF dataset not found: data/DrugLikeSMILSE-debug'

    tok = _ensure_special_tokens(SmilesTokenizer(vocab_file='vocab.txt'))
    CLS = tok.convert_tokens_to_ids(getattr(tok, 'cls_token', None))
    SEP = tok.convert_tokens_to_ids(getattr(tok, 'sep_token', None))
    BOS, EOS = _bos_eos_ids(tok)

    cache_dir = tmp_path.as_posix()
    ds = dataloader.get_dataset(
        dataset_name='smiles', tokenizer=tok, wrap=False, mode='train', cache_dir=cache_dir,
        block_size=64, num_proc=1, streaming=False, insert_eos=False, insert_special_tokens=False,
        raw_data_path=raw, no_special_tokens=True,
    )
    assert len(ds) > 0
    sample = ds[0]
    ids = sample['input_ids'].tolist()
    # 不应包含任何特殊 token
    for sid in [CLS, SEP, BOS, EOS]:
        if sid is not None:
            assert sid not in ids
    # attention_mask 合理且长度匹配
    assert sample['attention_mask'].shape[0] == 64
    # 缓存路径应包含 _noSpecials 标签
    expected = _expected_cache_path(cache_dir, wrap=False, insert_eos=False, insert_special_tokens=False, no_special_tokens=True)
    assert os.path.exists(expected), expected
