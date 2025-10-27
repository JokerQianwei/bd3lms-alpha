"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""

import csv
import functools
import logging
import math

import fsspec
import lightning
import torch
from timm.scheduler import CosineLRScheduler


def count_parameters(model):
  return sum(p.numel()
             for p in model.parameters()
             if p.requires_grad)

def fsspec_exists(filename):
  """Check if a file exists using fsspec."""
  fs, _ = fsspec.core.url_to_fs(filename)
  return fs.exists(filename)


def fsspec_listdir(dirname):
  """Listdir in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
  """Mkdirs in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
  if torch.isnan(tensor).any():
    print(name, tensor)

def update_and_save_csv(save_dict, csv_path):
  num_samples = len(save_dict['gen_ppl'])
  with fsspec.open(csv_path, 'a') as f:
    writer = csv.DictWriter(f, fieldnames=save_dict.keys())
    if fsspec_exists(csv_path) is False:
        writer.writeheader()
    for i in range(num_samples):
      row = {k: v[i] for k, v in save_dict.items()}
      writer.writerow(row)

class CosineDecayWarmupLRScheduler(
  CosineLRScheduler,
  torch.optim.lr_scheduler._LRScheduler):
  """Wrap timm.scheduler.CosineLRScheduler
  Enables calling scheduler.step() without passing in epoch.
  Supports resuming as well.
  Adapted from:
    https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_epoch = -1
    self.step(epoch=0)

  def step(self, epoch=None):
    if epoch is None:
      self._last_epoch += 1
    else:
      self._last_epoch = epoch
    # We call either step or step_update, depending on
    # whether we're using the scheduler every epoch or every
    # step.
    # Otherwise, lightning will always call step (i.e.,
    # meant for each epoch), and if we set scheduler
    # interval to "step", then the learning rate update will
    # be wrong.
    if self.t_in_epochs:
      super().step(epoch=self._last_epoch)
    else:
      super().step_update(num_updates=self._last_epoch)


class LoggingContext:
  """Context manager for selective logging."""
  def __init__(self, logger, level=None, handler=None, close=True):
    self.logger = logger
    self.level = level
    self.handler = handler
    self.close = close

  def __enter__(self):
    if self.level is not None:
      self.old_level = self.logger.level
      self.logger.setLevel(self.level)
    if self.handler:
      self.logger.addHandler(self.handler)

  def __exit__(self, et, ev, tb):
    if self.level is not None:
      self.logger.setLevel(self.old_level)
    if self.handler:
      self.logger.removeHandler(self.handler)
    if self.handler and self.close:
      self.handler.close()


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
  """Initializes multi-GPU-friendly python logger."""

  logger = logging.getLogger(name)
  logger.setLevel(level)

  # this ensures all logging levels get marked with the rank zero decorator
  # otherwise logs would get multiplied for each GPU process in multi-GPU setup
  for level in ('debug', 'info', 'warning', 'error',
                'exception', 'fatal', 'critical'):
    setattr(logger,
            level,
            lightning.pytorch.utilities.rank_zero_only(
              getattr(logger, level)))

  return logger


class Sampler:
  def __init__(self, shape):
    self.shape = shape

  def _sampling_noise(self):
    pass
  
  def _hard_sample(self, logits):
    pass

  def _soft_sample(self, logits):
    return 0

  def _process_logits(self, logits):
    return logits

  def sample(self, logits):
    logits = self._process_logits(logits)
    noise = self._sampling_noise()
    noise = noise[: logits.shape[0], :]
    logits = logits + noise.to(
      dtype=logits.dtype, device=logits.device)
    hard_sample = self._hard_sample(logits)
    soft_sample = self._soft_sample(logits)
    return soft_sample + (hard_sample - soft_sample).detach()


@functools.lru_cache(maxsize=None)
def log_n_choose_k(n, k):
  prior_loss = 0.0
  for i in range(k):
    prior_loss += (math.log(n - i) - math.log(k - i))
  return prior_loss


@functools.lru_cache(maxsize=None)
def log_n_permute_k(n, k):
  ans = 0.0
  for i in range(k):
    ans += math.log(n - i)
  return ans


class TopKSampler(Sampler):
  def __init__(self, k, shape, gamma_tau=1.0,
               noise_type='sog'):
    super().__init__(shape)
    self.k = k
    self.gamma_tau = gamma_tau
    self.num_betas = 10
    self.sampler = torch.distributions.gamma.Gamma(
      1 / k * torch.ones(self.num_betas, * self.shape), 1.0)
    self.noise_type = noise_type

  def _sampling_noise(self):
    if self.noise_type == 'sog':
      noise = self.sampler.sample()
      beta = self.k / torch.arange(1, self.num_betas + 1, 1,
                                  dtype=torch.float32)
      beta = beta[:, None, None, None]
      assert beta.ndim == noise.ndim
      s = noise / beta
      s = torch.sum(s, axis=0)
      s = s - math.log(self.num_betas)
      return self.gamma_tau * (s / self.k)
    elif self.noise_type == 'gumbel':
      return - (1e-10 - (torch.rand(* self.shape)
                         + 1e-10).log()).log()
    elif self.noise_type == 'deterministic':
      return torch.zeros(* self.shape)

  def _process_logits(self, logits):
    assert logits.ndim == 3
    return logits

  def _hard_sample(self, logits):
    thresholds, _ = torch.sort(logits, dim=-1)
    thresholds = thresholds[:, :, - self.k][:, :, None]
    return (logits >= thresholds).type(logits.dtype)

  def _soft_sample(self, logits):
    soft_top_k = logits - torch.mean(logits, dim=-1,
                                     keepdim=True)
    return soft_top_k / torch.norm(soft_top_k, dim=-1,
                                   keepdim=True)

class GaussianSampler:
  def __init__(self, constrain_logits):
    self.constrain_logits = constrain_logits

  def gaussian_params_from_logits(self, logits):
    assert logits.ndim == 3
    n = logits.shape[-1] // 2
    mu = logits[:, :, :n]
    log_var = logits[:, :, n:]
    if self.constrain_logits:
      # mu \in [0, 1]
      mu = torch.tanh(mu)
      # var \in [0, 1]
      # log_var \in (- \inf, 0]
      log_var = - torch.nn.functional.softplus(log_var)
    return mu, log_var

  def sample(self, x):
    mu, log_var = self.gaussian_params_from_logits(x)
    sigma = log_var.exp().sqrt()
    return mu + sigma * torch.randn_like(mu)

from tdc import Oracle, Evaluator
from typing import List
from rdkit import Chem

class SmilesMetrics:
    """
    - validity = 有效分子数 / n_target
    - uniqueness = 去重后有效分子数 / 有效分子数
    - diversity = TDC Evaluator('diversity') 于“有效且去重”的集合（少于 2 条记 0.0）
    - quality = 满足 (QED≥0.6 且 SA≤4.0) 的有效分子数 / n_target
    """
    def __init__(self, n_target: int):
        self.n_target = max(int(n_target), 1)
        self.evaluator = Evaluator('diversity')
        self.oracle_qed = Oracle('qed')
        self.oracle_sa = Oracle('sa')

    @staticmethod
    def _clean_smiles_text(s: str) -> str:
        """清洗：移除特殊标记 [BOS]/[EOS]/[PAD]；
        若出现 [EOS]，仅保留其之前的内容；最后去除空格并无缝拼接。"""
        if not isinstance(s, str):
            return ""
        # 截断到首个 [EOS]
        idx = s.find("[EOS]")
        if idx >= 0:
            s = s[:idx]
        # 去特殊标记（兼容存在或不存在空格的场景）
        for tok in ("[BOS]", "[EOS]", "[PAD]"):
            s = s.replace(tok, "")
        # 去空格并拼接
        return "".join(s.split())

    def compute(self, smiles_list: List[str]):
        """基于 RDKit 进行有效性判定，按 canonical SMILES 计算指标。"""
        smiles_list = smiles_list or []
        cleaned = [self._clean_smiles_text(s) for s in smiles_list]

        # RDKit 有效性 + 规范化 SMILES（canonical）
        mols = [Chem.MolFromSmiles(s) for s in cleaned]
        valid_smiles = [Chem.MolToSmiles(m, canonical=True) for m in mols if m is not None]

        valid_count = len(valid_smiles)
        validity = valid_count / self.n_target

        # 唯一性：在有效集合上去重
        unique_valid = list(set(valid_smiles))
        uniqueness = (len(unique_valid) / valid_count) if valid_count > 0 else 0.0

        # 多样性：需要至少两个不同分子
        diversity = float(self.evaluator(unique_valid)) if len(unique_valid) >= 2 else 0.0

        # 质量：仅对有效分子评估 QED/SA，再按阈值计数；分母为 n_target
        if valid_count > 0:
            qed_list = self.oracle_qed(valid_smiles)
            sa_list = self.oracle_sa(valid_smiles)
            quality_hits = sum(1 for q, s in zip(qed_list, sa_list) if (q >= 0.6 and s <= 4.0))
        else:
            quality_hits = 0
        quality = quality_hits / self.n_target

        return {
            "validity": float(validity),
            "uniqueness": float(uniqueness),
            "diversity": float(diversity),
            "quality": float(quality),
        }
        
