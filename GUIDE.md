# 记录实验进展

## 计划
- 统计 SMILES 的长度分布，确定 model.length, block-size=4(初步)】
  - block-size越小质量越高，越大速度越快 
- 弄懂原始模型是如何处理数据的，lm1b模型，是否开启 wrap


## 数据集
- 目录 data/DrugLikeSMILSE-12B-427M/train 总条数: 426_640_404
  - 4.27 亿分子, 12B token