"""
Arrow/ArrowDataset 检查工具

用途：
  - 直接查看 HuggingFace Datasets 保存的 Arrow 分片内容（.arrow 文件），或保存目录中的前若干条。
  - 自动打印 schema、预览若干行；若存在 `input_ids` 列，可选地用分词器解码查看文本。

用法示例：
  - 查看分片目录：
      python -u -m tools.arrow_inspect --path cache/smiles_cache/smiles_train_bs64_wrapped.dat --rows 5
  - 查看单个分片文件：
      python -u -m tools.arrow_inspect --path cache/smiles_cache/smiles_train_bs64_wrapped.dat/data-00000-of-00311.arrow --rows 5
  - 同时解码 input_ids 为文本（SMILES 词表）：
      python -u -m tools.arrow_inspect --path data/smiles_cache_len64/smiles_train_bs64_wrapped.dat  --rows 10 --decode-input-ids --tokenizer smiles --vocab vocab.txt

注意：
  - .dat 目录是 HF Datasets 的磁盘目录，内部包含若干 data-*-of-*.arrow 分片。
  - 对于 wrap=True 的缓存，常见列为：input_ids (list<int>), attention_mask (list<int>)。
  - 对于原始 SMILES 数据，常见列为：input (str) 等。
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, Iterable, List, Optional

import pyarrow as pa
import pyarrow.ipc as ipc

try:
    # 项目内 SMILES 分词器（可选）
    from tokenizer import SmilesTokenizer as RxnSmilesTokenizer  # type: ignore
except Exception:
    RxnSmilesTokenizer = None  # 在未使用 --tokenizer smiles 时不强制依赖

try:
    import transformers  # 用于通用解码（可选）
except Exception:
    transformers = None


def _open_arrow_batches(file_path: str) -> Iterable[pa.RecordBatch]:
    """以尽量兼容的方式打开单个 .arrow 文件，返回 RecordBatch 迭代器。"""
    with open(file_path, "rb") as f:
        # 优先尝试文件格式（RecordBatchFileReader）
        try:
            reader = ipc.open_file(f)
            for i in range(reader.num_record_batches):
                yield reader.get_batch(i)
            return
        except Exception:
            pass
        # 回退到流式格式（RecordBatchStreamReader）
        f.seek(0)
        reader = ipc.open_stream(f)
        for batch in reader:
            yield batch


def _collect_rows_from_batches(
    batches: Iterable[pa.RecordBatch],
    limit: int,
) -> List[Dict[str, object]]:
    """从一组 RecordBatch 中取前 limit 行，转换为 Python dict 列表。"""
    rows: List[Dict[str, object]] = []
    for batch in batches:
        # 将每列转为 Python 列表
        cols = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        batch_len = batch.num_rows
        for i in range(batch_len):
            rows.append({name: cols[name][i] for name in cols})
            if len(rows) >= limit:
                return rows
    return rows


def _find_arrow_files(path: str) -> List[str]:
    """接受目录或文件路径，返回待读取的 .arrow 文件列表（按名字排序）。"""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "data-*-of-*.arrow")))
        if not files:
            raise FileNotFoundError(f"目录下未发现 Arrow 分片：{path}")
        return files
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if not path.endswith(".arrow"):
            raise ValueError(f"指定的不是 .arrow 文件：{path}")
        return [path]


def _load_tokenizer(
    kind: Optional[str], tokenizer_name_or_path: Optional[str], vocab_path: Optional[str]
):
    if not kind:
        return None
    if kind == "smiles":
        if RxnSmilesTokenizer is None:
            raise RuntimeError("未找到项目内 SmilesTokenizer，确认本仓库可导入 tokenizer.py")
        vp = vocab_path or "vocab.txt"
        return RxnSmilesTokenizer(vocab_file=vp)
    # 其它：走 HF AutoTokenizer
    if transformers is None:
        raise RuntimeError("未安装 transformers，无法加载通用分词器")
    name_or_path = tokenizer_name_or_path or "bert-base-uncased"
    tok = transformers.AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def _maybe_truncate(seq: List, max_len: int = 32) -> str:
    if not isinstance(seq, (list, tuple)):
        return str(seq)
    if len(seq) <= max_len:
        return str(seq)
    head = seq[:max_len]
    return f"{head} ... (len={len(seq)})"


def inspect_arrow(
    path: str,
    rows: int = 5,
    decode_input_ids: bool = False,
    tokenizer_kind: Optional[str] = None,
    tokenizer_name_or_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    max_seq_print: int = 32,
) -> None:
    files = _find_arrow_files(path)

    # 读取首个分片以打印 schema
    first_batches = list(_open_arrow_batches(files[0]))
    if not first_batches:
        print("空的 Arrow 分片，未读取到任何 batch。")
        return
    schema = first_batches[0].schema
    print("Schema:")
    for i, field in enumerate(schema):
        print(f"  - {i}: {field.name}: {field.type}")

    # 收集前 rows 行（跨分片）
    collected: List[Dict[str, object]] = []
    for fp in files:
        batches = _open_arrow_batches(fp)
        needed = rows - len(collected)
        if needed <= 0:
            break
        collected.extend(_collect_rows_from_batches(batches, needed))
        if len(collected) >= rows:
            break

    print(f"\nPreview first {len(collected)} row(s):")
    for idx, row in enumerate(collected):
        print(f"\n# Row {idx}")
        for k, v in row.items():
            if isinstance(v, list):
                print(f"  {k}: {_maybe_truncate(v, max_seq_print)}")
            else:
                print(f"  {k}: {v}")

        # 尝试对 input_ids 做解码
        if decode_input_ids and "input_ids" in row and isinstance(row["input_ids"], list):
            tok = _load_tokenizer(tokenizer_kind, tokenizer_name_or_path, vocab_path)
            if tok is not None:
                try:
                    # transformers 及本项目分词器均支持 .decode(ids)
                    text = tok.decode(row["input_ids"])  # type: ignore[arg-type]
                    print(f"  decoded(input_ids): {text}")
                except Exception as e:
                    print(f"  decoded(input_ids) 失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect Arrow / HF Datasets shards")
    parser.add_argument("--path", required=True, help=".arrow 文件路径或包含分片的目录")
    parser.add_argument("--rows", type=int, default=5, help="预览的行数")
    parser.add_argument("--decode-input-ids", action="store_true", help="若存在 input_ids，则尝试用分词器解码")
    parser.add_argument("--tokenizer", choices=["smiles", "hf"], default=None, help="分词器类型：smiles 或 hf(AutoTokenizer)")
    parser.add_argument("--tokenizer-name-or-path", default=None, help="当 --tokenizer=hf 时使用的名称/路径")
    parser.add_argument("--vocab", dest="vocab_path", default=None, help="当 --tokenizer=smiles 时的词表文件路径")
    parser.add_argument("--max-seq-print", type=int, default=32, help="序列打印最大元素数（超出将截断）")
    args = parser.parse_args()

    inspect_arrow(
        path=args.path,
        rows=args.rows,
        decode_input_ids=args.decode_input_ids,
        tokenizer_kind=args.tokenizer,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        vocab_path=args.vocab_path,
        max_seq_print=args.max_seq_print,
    )


if __name__ == "__main__":
    main()
