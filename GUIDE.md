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

**参考训练指令**
```bash
export OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_NVLS_ENABLE=0
export NCCL_COLLNET_ENABLE=0
export NCCL_DEBUG=WARN

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_DIR=./runs PYTHONPATH=. \
python -u main.py \
model=large algo=bd3lm \
trainer.max_steps=330_000 \
loader.global_batch_size=640 loader.eval_global_batch_size=640 loader.num_workers=8 \
data.smiles_path=DATA/DrugLikeFragSeqV2-29B-425M-prepack-k20-n8 \
model.length=160 block_size=20 \
trainer.val_check_interval=0.1 \
trainer.limit_val_batches=0.01 \
trainer.devices=8 trainer.precision=bf16-mixed \
model.attn_backend=sdpa training.resample=True \
checkpointing.resume_from_ckpt=True \
checkpointing.resume_ckpt_path=/share/home/tm866079609100000/a875465180/yqw_bd3lms/bd3lms/runs/ex6_DrugLike_scratch_large_s160b20/20251014132103/runs/ex6_DrugLike_scratch_large_s160b20/20251014132103/checkpoints/0-150000.ckpt \
'hydra.run.dir=./runs/ex6_DrugLike_scratch_large_s${model.length}b${block_size}/${now:%Y%m%d%H%M%S}' \
'checkpointing.save_dir=${hydra:run.dir}' \
'sampling.logdir=${hydra:run.dir}/samples' \
'+wandb.mode=disabled' \
'+wandb.resume=never'

```


### SMILES训练
**max_stpes 推算**
- Train 数据量：保留 207_022_091 条
1 个 epoch = 模型在训练过程中 把这个训练集完整走一遍
每个 epoch 步数: steps_per_epoch = 总数据量 / global_batch_size
假设让每个模型看 5 遍完整的数据，即 5 个 epochs：
则：max_steps = steps_per_epoch x epochs_wanted
  =>  max_steps = (207022091/640)  x 1 ～  323_472 ~ 330_000

1个epoch的iter：214_754 ~ 220_000

参数量：
- 29B token 数据, 过滤条 20% ~ 23.2B token
- 一般来说一个参数对应10个token，那么 23.2B token，就需要对应 2B 参数量 -> 2000M
medium -> 322 M
large ->  651 M
large_1B -> 984M


```bash
python -u main.py \
    loader.global_batch_size=1600 \
    loader.eval_global_batch_size=1600 \
    model=large_1B \
    algo=mdlm \
    data=smiles \
    model.length=66 \
    wandb=None \
    trainer.val_check_interval=0.1 \
    trainer.limit_val_batches=0.05 \
    loader.num_workers=2 \
    data.raw_data_path=/share/home/tm866079609100000/a875465180/yqw_bd3lms/data/DrugLikeSMILSE-12B-427M \
    data.cache_dir=/share/home/tm866079609100000/a875465180/yqw_bd3lms/cache/cache-DrugLikeSMILES-12B-427M-addBOSEOS \
    '+wandb.mode=disabled' \
    '+wandb.resume=never'\
    model.attn_backend=sdpa \
    trainer.max_steps=220_000 \
    'hydra.run.dir=${hydra:runtime.cwd}/outputs/smiles/mdlm-addBOSEOS-len${model.length}/${now:%Y.%m.%d}/${now:%H%M%S}' \
    checkpointing.resume_from_ckpt=True \ 
    checkpointing.resume_ckpt_path=/share/home/tm866079609100000/a875465180/yqw_bd3lms/bd3lms-alpha/outputs/smiles/mdlm-addBOSEOS-len66/2025.10.24/142148/checkpoints/0-55000.ckpt


```

构造SMILES-cached数据：丢弃超长样本 448234/426640404 (0.11%); 实际训练的样本数量: 426_192_170
---
#### SMIELS实验记录
**第一次训练**
  | Name     | Type           | Params | Mode
----------------------------------------------------
0 | backbone | DIT            | 984 M  | train
1 | noise    | LogLinearNoise | 0      | train
----------------------------------------------------
Epoch 0:  10%|██████████                                                                                           | 26637/266371 [3:15:36<29:20:25,  2.27it/s, v_num=0]Epoch 0, global step 26637: 'val/nll' reached 0.98462 (best 0.98462), saving model to '/share/home/tm866079609100000/a875465180/yqw_bd3lms/bd3lms-alpha/outputs/smiles/mdlm-addBOSEOS-len66/2025.10.23/201851/checkpoints/best.ckpt' as top 1
Epoch 0:  20%|████████████████████▌                                                                                  | 53274/266371 [6:31:24<26:05:39,  2.27it/s, v_num=0]
Epoch 0, global step 53274: 'val/nll' was not in top 1
第二次恢复训练 smiles 从 step=55000 开始





### Fragment 训练
数据路径：/data/yqw/bd3lms/DATA/DrugLikeFragSeqV2-29B-425M

```bash
python -u main.py \
    loader.global_batch_size=1600 \
    loader.eval_global_batch_size=1600 \
    model=large_1B \
    algo=mdlm \
    data=smiles \
    model.length=66 \
    wandb=None \
    trainer.val_check_interval=0.1 \
    trainer.limit_val_batches=0.05 \
    loader.num_workers=2 \
    data.raw_data_path=/share/home/tm866079609100000/a875465180/yqw_bd3lms/data/DrugLikeSMILSE-12B-427M \
    data.cache_dir=/share/home/tm866079609100000/a875465180/yqw_bd3lms/cache/cache-DrugLikeSMILES-12B-427M-addBOSEOS \
    '+wandb.mode=disabled' \
    '+wandb.resume=never' \
    model.attn_backend=sdpa \
    trainer.max_steps=220_000 \
    'hydra.run.dir=${hydra:runtime.cwd}/outputs/smiles/mdlm-addBOSEOS-len${model.length}/${now:%Y.%m.%d}/${now:%H%M%S}' \
    checkpointing.resume_from_ckpt=True \
    checkpointing.resume_ckpt_path=/share/home/tm866079609100000/a875465180/yqw_bd3lms/bd3lms-alpha/outputs/smiles/mdlm-addBOSEOS-len66/2025.10.24/142148/checkpoints/0-55000.ckpt
```

Fragment 丢弃超长样本： 81833255/425438898 (19.24%); 实际训练的样本数量: 343_605_643

####  Fragment 实验记录
  | Name     | Type           | Params | Mode
----------------------------------------------------
0 | backbone | DIT            | 984 M  | train
1 | noise    | LogLinearNoise | 0      | train
----------------------------------------------------
Epoch 0:  10%|██████████                                                                                           | 21475/214754 [2:41:15<24:11:24,  2.22it/s, v_num=0]Epoch 0, global step 21475: 'val/nll' reached 0.76714 (best 0.76714), saving model to '/share/home/tm866079609100000/a875465180/yqw_bd3lms/bd3lms-alpha/outputs/fragment/mdlm-addBOSEOS--len66/2025.10.23/201916/checkpoints/best.ckpt' as top 1
Epoch 0:  20%|████████████████████▌                                                                                  | 42950/214754 [5:22:17<21:29:10,  2.22it/s, v_num=0]
Epoch 0, global step 42950: 'val/nll' reached 0.76419 (best 0.76419), saving model to '/share/home/tm866079609100000/a875465180/yqw_bd3lms/bd3lms-alpha/outputs/fragment/mdlm-addBOSEOS--len66/2025.10.23/201916/checkpoints/best.ckpt' as top 1
Epoch 0:  25%|█████████████████████████▉                                                                             | 53955/214754 [6:43:42<20:03:08,  2.23it/s, v_num=0Connection 
第二次恢复训练 fragment 从 step=50000 开始

---
## 采样
first_hitting 不支持并行采样
```bash
python -u -m main \
    mode=sample_eval \
    sampling.num_sample_batches=100 \
    loader.eval_batch_size=1 \
    data=smiles \
    algo=mdlm \
    algo.T=5000 \
    model=large_1B \
    model.length=66 \
    +eval.checkpoint_path=/share/home/tm866079609100000/a875465180/yqw_bd3lms/bd3lms-alpha/outputs/smiles/mdlm-addBOSEOS-len66/2025.10.23/201851/checkpoints/0-55000.ckpt \
    seed=2 \
    sampling.nucleus_p=0.9 \
    sampling.logdir=$PWD/sample_logs/samples_mdlm_len64 \
    model.attn_backend=sdpa \
    sampling.first_hitting=true

```




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