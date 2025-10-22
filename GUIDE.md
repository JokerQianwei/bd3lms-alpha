# 记录实验进展

## 计划
- 统计 SMILES 的长度分布，确定 model.length, block-size=4(初步)】
  - block-size 越小质量越高（越接近自回归），越大速度越快（越接近DDPM） 
  - block-size 是块的长度
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
sent1 [SEP] sent2-fragment [SEP]
[SEP] sent2-fragment [EOS] s [SEP]

特殊词元不匹配: SmilesTokenizer 默认的特殊词元是 [CLS], [SEP], [PAD] 等，这是BERT的习惯。而Block Diffusion代码库在 _group_texts 中明确使用了 [BOS] 和 [EOS]。如果您的 vocab.txt 中没有这两个词元，或者分词器没有正确地将它们识别为 bos_token 和 eos_token，那么在数据处理时可能会出错或产生非预期的行为。


## 测试 mdlm 部分快的代码是否正确
分别使用 高类药性的 smile 和 fragemnt 进行训练

### 生成数据cache
不添加任何额外的token，最大长度设置为64，大于的直接截断，并打印截断多少

**SMILES**
```bash
python -u main.py \
    loader.global_batch_size=100 \
    loader.eval_global_batch_size=100 \
    model=small \
    algo=mdlm \
    data=smiles \
    model.length=64 \
    wandb.name=mdlm-smiles \
    trainer.val_check_interval=1.0 \
    algo.ignore_bos=false \
    loader.global_batch_size=800 \
    loader.eval_global_batch_size=800 \
    data.raw_data_path=/share/home/tm866079609100000/a866071650/doomx/MolGen/DATA/DrugLikeSMILSE-12B-427M \
    data.cache_dir=/share/home/tm866079609100000/a875465180/yqw_bd3lms/cache/cache-DrugLikeSMILSE-12B-427M 
```
**Fragment**
数据路径：/data/yqw/bd3lms/DATA/DrugLikeFragSeqV2-29B-425M

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

python -u main.py \
    loader.global_batch_size=800 \
    loader.eval_global_batch_size=800 \
    model=small \
    algo=mdlm \
    data=smiles \
    model.length=64 \
    wandb.name=mdlm-smiles \
    trainer.val_check_interval=1.0 \
    algo.ignore_bos=false \
    loader.global_batch_size=800 \
    loader.eval_global_batch_size=800 \
    loader.num_workers=4 \
    data.raw_data_path=/share/home/tm866079609100000/a875465180/yqw_bd3lms/DrugLikeFragSeqV2-29B-425M \
    data.cache_dir=/share/home/tm866079609100000/a875465180/yqw_bd3lms/cache/cache-DrugLikeFragSeqV2-29B-425M
```
Fragment 丢弃超长样本： 81833255/425438898 (19.24%); 实际训练的样本数量: 343605643


---

## 新的思路 生成相似的分子
使用纯的 diffuion

- 关键：block_size 参数
  - block_size = 1, 即块的长度为1，退化成自回归
  - block_size = model.length, 块的长度等于序列长度，即一块，不分块，也就是纯的diffusion

- 目的：能生成和输入到 SMILES 性质类似的 SMILES
- 实现：
  - 数据：
    - 根据分子指纹聚类数据，一个SMILES可能出现在不同的序列中，
    - 一条序列中,包含很多条SMILES，一条序列都是具有相同特性的。初步设定，一条序列的长度统一为 4080，smiles直接用eos分割
    - 最终的训练数据就是，每条样本都包含了多条SMILES，这些SMLES具有相同的分子指纹特性。用 eos 隔开
  - 训练：
    - 将原项目中的 block_size = model.length，块的长度等于序列长度，即一块，不分块，也就是纯的diffusion
    - 使用 dit
  - 采样：
    - 如果我想生成和 SMILES A 类似的分子，我在采样的时候，将 A 固定在最前面的token上，然后扩散还原，其他token
    - 期望可以生成和A类似性质的SMLES
    - 然后根据EOS切割解码