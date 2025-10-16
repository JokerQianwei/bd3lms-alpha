import os
import sys


class _Cfg:
    class _Data:
        tokenizer_name_or_path = 'smiles'
        smiles_vocab_path = os.path.join(os.getcwd(), 'vocab.txt')
    data = _Data()


def main():
    # 确保可以从项目根目录导入 dataloader
    repo_root = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import dataloader

    # 1) 获取 SMILES 分词器（基于本地 vocab.txt）
    cfg = _Cfg()
    tokenizer = dataloader.get_tokenizer(cfg)

    # 构造两段简短 SMILES 文本，模拟 wrap 前的 token 列表
    texts = [
        'CC(=O)O',
        'c1ccccc1',
    ]
    tokenized = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
    examples = {'input_ids': tokenized['input_ids']}

    # 设置一个较小的 block_size，便于快速拼接
    block_size = 16

    # 2) 旧逻辑（错误）：用 encode() 默认 add_special_tokens=True 取特殊符号 id
    wrong_EOS = tokenizer.encode(tokenizer.eos_token)[0]
    wrong_BOS = tokenizer.encode(tokenizer.bos_token)[0]
    red = dataloader._group_texts(examples, block_size, wrong_BOS, wrong_EOS, insert_special_tokens=True)
    red_decoded = tokenizer.decode(red['input_ids'][0])

    # 期望“红”：出现 [CLS]，即 [BOS] 未被正确注入
    red_ok = ('[CLS]' in red_decoded) and ('[BOS]' not in red_decoded)

    # 3) 新逻辑（正确）：直接通过 convert_tokens_to_ids 获取 [BOS]/[EOS]
    right_EOS = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    right_BOS = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
    green = dataloader._group_texts(examples, block_size, right_BOS, right_EOS, insert_special_tokens=True)
    green_decoded = tokenizer.decode(green['input_ids'][0])

    # 期望“绿”：不含 [CLS]，且以 [BOS] 开头
    green_ok = ('[CLS]' not in green_decoded) and green_decoded.strip().startswith('[BOS]')

    print('RED check decoded:', red_decoded)
    print('GREEN check decoded:', green_decoded)

    if not red_ok:
        print('RED stage did not reproduce the bug.', file=sys.stderr)
        return 1
    if not green_ok:
        print('GREEN stage did not validate the fix.', file=sys.stderr)
        return 2

    print('OK: 先红后绿 ✅')
    return 0


if __name__ == '__main__':
    sys.exit(main())
