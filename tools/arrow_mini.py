import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 确保可以导入项目根目录下的 tokenizer.py
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tokenizer import SmilesTokenizer


file_path = '/data/yqw/bd3lms/DATA/DrugLikeFragSeqV2-29B-425M/train/data-00000-of-00099.arrow'
vocab_path = os.path.join(PROJECT_ROOT, 'vocab.txt')
tokenizer = SmilesTokenizer(vocab_file=vocab_path)
tokenizer.bos_token = "[BOS]"
tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
tokenizer.eos_token = "[EOS]"
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
tokenizer.sep_token = "[SEP]"
tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")

with open(file_path, "rb") as f:
    try:
        reader = ipc.open_file(f)
    except pa.ArrowInvalid:
        f.seek(0) # 必须将文件指针移回开头
        reader = ipc.open_stream(f)
    df = reader.read_all().to_pandas()

print(df.head())

text = '[*+]NC[SEP][*+]CC[*-][SEP][*+][C@@H]1C([*-])CCC1O[SEP][*+]C(=O)S[*-][SEP][*-]c1ccccc1'
ids = tokenizer(text, add_special_tokens=False, padding=False, truncation=False, return_attention_mask=False)
print(ids)