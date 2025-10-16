import glob
import os

import pyarrow.ipc as ipc

from tokenizer import SmilesTokenizer


def read_some_smiles(arrow_dir, n):
    """从 Arrow 分片目录读取前 n 条 SMILES（列名 'input'）。"""
    shard_paths = sorted(glob.glob(os.path.join(arrow_dir, "data-*-of-*.arrow")))
    smiles = []
    for shard in shard_paths:
        with open(shard, "rb") as f:
            reader = ipc.open_stream(f)
            for batch in reader:
                col = batch.column("input")
                values = col.to_pylist()
                for v in values:
                    if v is not None:
                        smiles.append(v)
                        if len(smiles) >= n:
                            return smiles
    return smiles


data_dir = "data/DrugLikeSMILSE-12B-427M/train"
vocab_path = "vocab.txt"
n = 10

smiles = read_some_smiles(data_dir, n)
print(smiles)
tok = SmilesTokenizer(vocab_path)

for s in smiles[:n]:
    tokens = tok._tokenize(s)
    ids = tok.encode(s, add_special_tokens=True)
    print("SMILES:", s)
    print("TOKENS:", tokens)
    print("IDS:", ids)
    print("——")


"""
python -m scripts.smiles_tokenizer_smoke
"""