import os
import math
import torch
from torch import Tensor
import torch.nn.functional as F
import torchmetrics
import typing
from typing import Union
from tqdm import tqdm
import transformers

LOG2 = torch.log(torch.tensor(2.0))

class NLL(torchmetrics.aggregation.MeanMetric): ...
class NFEs(torchmetrics.aggregation.MeanMetric): ...

class BPD(NLL):
  def compute(self) -> Tensor:
    return self.mean_value / self.weight / LOG2

class Perplexity(NLL):
  def compute(self) -> Tensor:
    return torch.exp(self.mean_value / self.weight)

class Metrics:
  def __init__(self, config=None) -> None:
    self.config = config
    metrics = torchmetrics.MetricCollection({'nll': NLL(), 'bpd': BPD(), 'ppl': Perplexity()})

    # context length
    self.block_size = getattr(config, 'block_size', getattr(config.model, 'length', 1024))

    # train/val aggregators
    self.nfes = NFEs()
    self.train_nlls = metrics.clone(prefix='train/')
    self.valid_nlls = metrics.clone(prefix='val/')
    self.gen_ppl = Perplexity()
    self.gen_entropy = NLL()
    self.gen_ppls, self.gen_nfes, self.gen_entropies, self.gen_lengths = [], [], [], []

    # algo vars
    self.sampling_eps = config.training.sampling_eps
    if getattr(config.algo, 'clip_search_delta', None):
      self.clip_search_delta = config.algo.clip_search_delta
    self.valid_vars = {self.sampling_eps: []}
    if getattr(config.algo, 'var_min', None):
      self.valid_vars = self.init_valid_vars()

    # ---- Eval / HF 离线相关开关（新增：都带默认值） ----
    self.eval_ppl_batch_size = getattr(config.eval, 'perplexity_batch_size', 8)
    self.gen_ppl_enabled = getattr(config.eval, 'gen_ppl_enabled', True)
    self.local_files_only = getattr(config.eval, 'local_files_only', True)
    self.gen_ppl_eval_model_name_or_path = getattr(
      config.eval, 'gen_ppl_eval_model_name_or_path', None
    )

    # 懒加载：延后到第一次用 PPL 时再加载
    self._tokenizer = None
    self._eval_model = None

  def init_valid_vars(self):
    eps = self.sampling_eps
    if self.block_size > 1:
      self.valid_vars = {(eps, 1): []}
      for width in self.config.algo.clip_search_widths:
        for i in torch.arange(0, 1 - width + self.clip_search_delta, self.clip_search_delta):
          mn = torch.clamp(i, min=self.sampling_eps).item()
          mx = torch.clamp(i + width, min=self.sampling_eps).item()
          self.valid_vars[(mn, mx)] = []
    else:
      self.valid_vars = {(eps, 1): [], (1, 1): []}

  def to(self, *args, **kwargs):
    self.train_nlls = self.train_nlls.to(*args, **kwargs)
    self.valid_nlls = self.valid_nlls.to(*args, **kwargs)
    self.gen_ppl = self.gen_ppl.to(*args, **kwargs)
    self.nfes = self.nfes.to(*args, **kwargs)
    self.gen_entropy = self.gen_entropy.to(*args, **kwargs)
    # eval model/ tokenizer 若已加载，也迁移
    if self._eval_model is not None:
      self._eval_model = self._eval_model.to(*args, **kwargs)

  def reset(self):
    self.gen_ppls, self.gen_nfes, self.gen_entropies, self.gen_lengths = [], [], [], []
    self.train_nlls.reset()
    self.valid_nlls.reset()
    self.gen_ppl.reset()
    self.gen_entropy.reset()
    self.nfes.reset()
    if getattr(self.config.algo, 'var_min', None):
      self.init_valid_vars()

  # ---------- 懒加载与离线安全封装 ----------
  def _ensure_tokenizer(self):
    if self._tokenizer is not None:
      return self._tokenizer
    if not self.gen_ppl_enabled or not self.gen_ppl_eval_model_name_or_path:
      raise RuntimeError(
        "Generative PPL 被禁用或未提供本地模型路径（config.eval.gen_ppl_eval_model_name_or_path）。"
      )

    # 强制本地：禁网环境下必须
    try:
      tok = transformers.AutoTokenizer.from_pretrained(
        self.gen_ppl_eval_model_name_or_path,
        local_files_only=self.local_files_only
      )
    except Exception as e:
      raise RuntimeError(
        f"无法本地加载 tokenizer：{self.gen_ppl_eval_model_name_or_path}。\n"
        f"- 请将评估模型下载到本地目录并填入该路径，或\n"
        f"- 在配置中设定 eval.gen_ppl_enabled=False 以跳过 PPL。\n"
        f"原始错误：{e}"
      )
    if tok.pad_token is None:
      tok.pad_token = tok.eos_token
      tok.pad_token_id = tok.eos_token_id
    self._tokenizer = tok
    return tok

  def _ensure_eval_model(self, device='cuda'):
    if self._eval_model is not None:
      return self._eval_model
    if not self.gen_ppl_enabled or not self.gen_ppl_eval_model_name_or_path:
      raise RuntimeError(
        "Generative PPL 被禁用或未提供本地模型路径（config.eval.gen_ppl_eval_model_name_or_path）。"
      )
    try:
      model = transformers.AutoModelForCausalLM.from_pretrained(
        self.gen_ppl_eval_model_name_or_path,
        local_files_only=self.local_files_only
      ).eval()
    except Exception as e:
      raise RuntimeError(
        f"无法本地加载评估模型：{self.gen_ppl_eval_model_name_or_path}。\n"
        f"- 请提供包含 config.json / tokenizer / 权重的本地目录，或\n"
        f"- 设定 eval.gen_ppl_enabled=False 跳过 PPL。\n"
        f"原始错误：{e}"
      )
    if 'llama2' not in str(self.gen_ppl_eval_model_name_or_path).lower():
      model = model.to(device)
    self._eval_model = model
    return model

  @torch.no_grad()
  def _eval_retokenize(self, text_samples, max_length, device):
    # 只有真正需要时才会触发 tokenizer 加载
    tokenizer = self._ensure_tokenizer()
    name = str(self.gen_ppl_eval_model_name_or_path).lower()
    if 'llama2' in name:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = tokenizer(text_samples, **tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in name:
      attn_mask = attn_mask.to(device)
      samples = samples.to(device)
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def record_generative_perplexity(
    self,
    text_samples: typing.List[str],
    max_length: int,
    batch_size: Union[int, None] = None,
    retokenize: bool = True,
    stride: int = 512,
    device: str = 'cuda'
  ) -> None:
    # 若关闭或无模型，直接跳过
    if not self.gen_ppl_enabled or not self.gen_ppl_eval_model_name_or_path:
      return

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    eval_model = self._ensure_eval_model(device=device)

    # 样本编码
    if retokenize:
      samples, attn_mask, eval_context_size = self._eval_retokenize(
        text_samples, max_length=max_length, device=device
      )
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(device)
      eval_context_size = samples.shape[-1]

    if batch_size is None:
      batch_size = min(self.eval_ppl_batch_size, samples.shape[0])

    num_batches = samples.shape[0] // batch_size
    for b in range(num_batches):
      samples_batch = samples[b * batch_size: (b + 1) * batch_size]
      attn_mask_batch = attn_mask[b * batch_size: (b + 1) * batch_size]

      nlls_accum = torch.zeros_like(samples_batch, dtype=torch.float32)
      valid_tokens_accum = torch.zeros_like(samples_batch, dtype=torch.float32)

      num_strides = max(math.ceil((samples_batch.shape[-1] - eval_context_size + stride) / stride), 1)

      for i in tqdm(range(num_strides), desc='Sliding Window Gen PPL'):
        if i == 0:
          start = 0
          end = min(eval_context_size, samples_batch.shape[-1])
        else:
          start = i * stride
          end = min(start + eval_context_size, samples_batch.shape[-1])

        sample_chunk = samples_batch[..., start:end]
        attn_mask_chunk = attn_mask_batch[..., start:end]

        logits = eval_model(sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)

        nlls = F.cross_entropy(logits[..., :-1], sample_chunk[..., 1:], reduction='none')
        valid_tokens = (sample_chunk[..., 1:] != self._ensure_tokenizer().eos_token_id).to(torch.float)

        if i == 0:
          nlls_accum[..., start + 1:end] += nlls
          valid_tokens_accum[..., start + 1:end] += valid_tokens
        else:
          update_start = (start + eval_context_size - stride)
          update_window = end - update_start
          nlls_accum[..., update_start:end] += nlls[..., -update_window:]
          valid_tokens_accum[..., update_start:end] += valid_tokens[..., -update_window:]

      # gen ppl
      avg_nll = (nlls_accum * valid_tokens_accum).sum() / valid_tokens_accum.sum()
      self.gen_ppls.append(avg_nll.exp().detach().cpu().item())
      self.gen_ppl.update(nlls_accum, valid_tokens_accum)

      # entropy（样本 token 频率的香农熵）
      entropy_full = 0
      for i in range(samples_batch.shape[0]):
        _, counts = torch.unique(samples_batch[i], return_counts=True, sorted=False)
        entropy = torch.special.entr(counts.float() / counts.sum()).sum()
        entropy_full += entropy
      self.gen_entropies.append(entropy_full.detach().cpu().item())
      self.gen_entropy.update(entropy_full, samples_batch.shape[0])

      self.gen_lengths.append(valid_tokens_accum.sum().detach().cpu().item())
