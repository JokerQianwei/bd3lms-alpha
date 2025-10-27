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

import numpy as np
from tdc import Oracle, Evaluator
from typing import List

try:
    # RDKit 用于有效性判断与规范化（canonical SMILES）
    from rdkit import Chem  # type: ignore
    _HAS_RDKIT = True
except Exception:
    Chem = None  # type: ignore
    _HAS_RDKIT = False

class SmilesMetrics:
    """SMILES 评测：validity/uniqueness/diversity/quality。

    关键修正：
    - 先对解码字符串做清洗（去空格与特殊标记），避免 RDKit 解析失败；
    - 使用 RDKit 过滤无效分子并做 canonical 化，再计算各指标；
    - 对可能的 NaN/异常做稳健回退（置 0.0）。
    """

    # 仅移除明确的特殊标记，不移除诸如 [nH] 等合法 SMILES 原子标记
    SPECIAL_TOKENS = {"[BOS]", "[EOS]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}

    def __init__(self, n_target: int):
        self.n_target = max(int(n_target), 1)
        self.evaluator = Evaluator('diversity')
        self.oracle_qed = Oracle('qed')
        self.oracle_sa = Oracle('sa')

    @staticmethod
    def _clean_smiles_text(s: str) -> str:
        """将解码后的字符串清洗为 SMILES：
        - 按空格切分去除特殊标记；
        - 其余 token 无缝拼接（不引入空格）。
        """
        if not isinstance(s, str):
            return ""
        tokens = s.strip().split()
        # 若包含分隔符 [EOS]，仅取第一段，避免拼接跨多条 SMILES
        if "[EOS]" in tokens:
            tokens = tokens[: tokens.index("[EOS]")]
        filtered: List[str] = [t for t in tokens if t not in SmilesMetrics.SPECIAL_TOKENS]
        return "".join(filtered)

    def _valid_canonical_all(self, smiles: List[str]) -> List[str]:
        """返回“有效且 canonical 化”的列表（保留重复项）。

        若 RDKit 不可用，则仅做非空过滤。
        """
        cleaned = [self._clean_smiles_text(s) for s in smiles]
        if _HAS_RDKIT:
            valid_all: List[str] = []
            for s in cleaned:
                if not s:
                    continue
                try:
                    mol = Chem.MolFromSmiles(s)
                except Exception:
                    mol = None
                if mol is None:
                    continue
                try:
                    cano = Chem.MolToSmiles(mol, canonical=True)
                except Exception:
                    cano = s
                valid_all.append(cano)
        else:
            valid_all = [s for s in cleaned if s]
        return valid_all

    @staticmethod
    def _dedup_preserve_order(items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def compute(self, smiles_list):
        """计算四个指标。

        口径：
        - validity = 有效分子数 / n_target
        - uniqueness = 去重后有效分子数 / 有效分子数
        - diversity = TDC Evaluator('diversity') 于“有效且去重”的集合；不足 2 条或 NaN 时记 0.0
        - quality = 满足 (QED≥0.6 且 SA≤4.0) 的有效分子数 / n_target
        """
        smiles_list = smiles_list or []

        # 有效（含重复）与去重后的集合
        valid_all = self._valid_canonical_all(smiles_list)
        uniq_valid = self._dedup_preserve_order(valid_all)
        num_valid = len(valid_all)

        # validity：化学有效率，以 n_target 为分母
        validity = num_valid / self.n_target

        # uniqueness：在“有效分子”范围内的去重率（去重后 / 含重复）
        uniqueness = (len(uniq_valid) / max(len(valid_all), 1)) if valid_all else 0.0

        # diversity：对“有效且去重”的集合
        if len(uniq_valid) > 1:
            try:
                diversity = float(self.evaluator(uniq_valid))
                if not np.isfinite(diversity):
                    diversity = 0.0
            except Exception:
                diversity = 0.0
        else:
            diversity = 0.0

        # quality：在“有效分子”上跑 QED/SA，再与阈值比较；分母仍然用 n_target
        ok = 0
        if num_valid > 0:
            try:
                q_list = self.oracle_qed(valid_all)
                s_list = self.oracle_sa(valid_all)
                # 防御性：过滤非数
                for qq, ss in zip(q_list, s_list):
                    if qq is None or ss is None:
                        continue
                    try:
                        qv = float(qq)
                        sv = float(ss)
                    except Exception:
                        continue
                    if np.isfinite(qv) and np.isfinite(sv) and (qv >= 0.6) and (sv <= 4.0):
                        ok += 1
            except Exception:
                ok = 0
        quality = ok / self.n_target

        return {
            "validity": float(validity),
            "uniqueness": float(uniqueness),
            "diversity": float(diversity),
            "quality": float(quality),
        }
    
