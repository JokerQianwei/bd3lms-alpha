# 记录实验进展

## 计划
- 统计 SMILES 的长度分布，确定 model.length, block-size=4(初步)】
  - block-size越小质量越高，越大速度越快 
- 弄懂原始模型是如何处理数据的，lm1b模型，是否开启 wrap


## 数据集
- 目录 data/DrugLikeSMILSE-12B-427M/train 总条数: 426_640_404
  - 4.27 亿分子, 12B token

**数据长度统计**
共统计 20000000 条样本，用时 2021.7s。示例长度：[26, 32, 31, 29, 48, 27, 33, 44, 53, 34]
SMILES Token ID 序列长度统计（+1 模拟 EOS）：
count    2.000000e+07
mean     3.692242e+01
std      8.881035e+00
min      5.000000e+00
25%      3.100000e+01
50%      3.700000e+01
75%      4.300000e+01
max      1.150000e+02
dtype: float64
50% 的SMILES序列长度 (tokenized) 小于: 37.0
75% 的SMILES序列长度 (tokenized) 小于: 43.0
90% 的SMILES序列长度 (tokenized) 小于: 48.0
95% 的SMILES序列长度 (tokenized) 小于: 52.0
99% 的SMILES序列长度 (tokenized) 小于: 58.0
99% 的SMILES序列长度 (tokenized) 小于: 64.0
已保存直方图：outputs/smiles_length_hist.png


For wrapped batches:
[BOS] sent1 [EOS] sent2-fragment [EOS]
[BOS] sent2-fragment [EOS] sent3 [EOS]

特殊词元不匹配: SmilesTokenizer 默认的特殊词元是 [CLS], [SEP], [PAD] 等，这是BERT的习惯。而Block Diffusion代码库在 _group_texts 中明确使用了 [BOS] 和 [EOS]。如果您的 vocab.txt 中没有这两个词元，或者分词器没有正确地将它们识别为 bos_token 和 eos_token，那么在数据处理时可能会出错或产生非预期的行为。

## 模型
- 训练两个模型
  - 一个使用 warp
  - 一个不使用 warp