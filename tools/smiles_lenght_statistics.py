import glob
import os
import sys
import time
from typing import Iterator, List, Optional

import pandas as pd
import pyarrow.ipc as ipc

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tokenizer import SmilesTokenizer


# 硬编码配置（如需修改，直接改这里）
DATA_DIR = "data/DrugLikeSMILSE-12B-427M/train"
VOCAB_PATH = "vocab.txt"
MAX_ROWS: Optional[int] = 20_000_000  # 例如设为 200_000 做快速统计；为 None 表示全量
BATCH_SIZE = 8192
PRINT_EVERY = 100000  # 进度打印频率（条）


def stream_smiles(arrow_dir: str, max_rows: Optional[int] = None) -> Iterator[str]:
    """逐条流式读取 SMILES（列名 'input'）。

    - 目录需包含形如 data-00000-of-*.arrow 的文件。
    - 使用 Arrow 流式读取，避免一次性加载内存。
    """
    shard_paths = sorted(glob.glob(os.path.join(arrow_dir, "data-*-of-*.arrow")))
    if not shard_paths:
        raise FileNotFoundError(f"未找到 Arrow 分片: {arrow_dir}/data-*-of-*.arrow")

    emitted = 0
    total_shards = len(shard_paths)
    for shard_idx, shard in enumerate(shard_paths, start=1):
        with open(shard, "rb") as f:
            reader = ipc.open_stream(f)
            for batch in reader:
                col = batch.column("input")
                values: List[Optional[str]] = col.to_pylist()
                for v in values:
                    if v is None:
                        continue
                    yield v
                    emitted += 1
                    if emitted % PRINT_EVERY == 0:
                        print(f"[读取进度] 分片 {shard_idx}/{total_shards}，累计 {emitted} 条…", flush=True)
                    if max_rows is not None and emitted >= max_rows:
                        return


def main():
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(
            f"未找到词表文件：{VOCAB_PATH}。请在仓库根目录放置 vocab.txt，或修改 VOCAB_PATH。"
        )

    print(f"加载分词器：{VOCAB_PATH}")
    tok = SmilesTokenizer(VOCAB_PATH)
    # 与仓库风格一致：BERT 风格的 BOS/EOS（虽本统计不使用 special tokens）
    tok.bos_token = tok.cls_token
    tok.eos_token = tok.sep_token

    print(f"从 {DATA_DIR} 读取 SMILES（max_rows={MAX_ROWS}）…")
    lengths: List[int] = []

    batch: List[str] = []
    processed = 0
    t0 = time.time()
    for s in stream_smiles(DATA_DIR, MAX_ROWS):
        batch.append(s)
        if len(batch) >= BATCH_SIZE:
            outs = tok(batch, add_special_tokens=False, return_attention_mask=False,
                       return_token_type_ids=False)
            for ids in outs["input_ids"]:
                lengths.append(len(ids) + 1)  # 手动模拟添加 [EOS]
            batch.clear()
            processed += len(outs["input_ids"])
            if processed % PRINT_EVERY < BATCH_SIZE:  # 约每 PRINT_EVERY 条打印一次
                dt = time.time() - t0
                speed = processed / dt if dt > 0 else 0
                print(f"[分词进度] 已处理 {processed} 条，速度 {speed:.1f} 样本/秒", flush=True)

    # 处理尾批
    if batch:
        outs = tok(batch, add_special_tokens=False, return_attention_mask=False,
                   return_token_type_ids=False)
        for ids in outs["input_ids"]:
            lengths.append(len(ids) + 1)
        batch.clear()
        processed += len(outs["input_ids"])

    if not lengths:
        raise RuntimeError("未统计到任何序列长度，请检查数据目录与列名是否正确（应为 'input'）。")

    dt = time.time() - t0
    # 明确打印目录下总条数（等同于本次处理样本数；若设置了 MAX_ROWS，为被截断的数量）
    print(f"目录 {DATA_DIR} 总条数（本次统计覆盖）: {processed}")
    print(f"共统计 {len(lengths)} 条样本，用时 {dt:.1f}s。示例长度：{lengths[:10]}")
    ser = pd.Series(lengths, dtype="int32")

    print("SMILES Token ID 序列长度统计（+1 模拟 EOS）：")
    print(ser.describe())

    quantiles = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    for q in quantiles:
        print(f"{int(q*100)}% 的SMILES序列长度 (tokenized) 小于: {ser.quantile(q)}")

    # 直方图：在无图形环境下尝试保存到 outputs/
    try:
        import matplotlib.pyplot as plt  # type: ignore

        os.makedirs("outputs", exist_ok=True)
        ser.hist(bins=50)
        plt.title("Distribution of SMILES Token Sequence Lengths (+1 for EOS)")
        plt.xlabel("Sequence Length (after tokenization)")
        plt.ylabel("Frequency")
        out_path = os.path.join("outputs", "smiles_length_hist.png")
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"已保存直方图：{out_path}")
        try:
            plt.show()
        except Exception:
            pass
    except Exception as e:
        print(f"绘图失败（可忽略）：{e}")


if __name__ == "__main__":
    main()
