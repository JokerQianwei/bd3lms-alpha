from typing import List
from rdkit import Chem
from tdc import Oracle, Evaluator


class SmilesMetrics:
    """
    SMILES 评测（独立模块）：基于 RDKit 判定有效性，按 canonical SMILES 计算四项指标。

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
        """清洗：移除 [BOS]/[EOS]/[PAD]；若出现 [EOS]，仅保留其之前内容；去空格拼接。"""
        if not isinstance(s, str):
            return ""
        idx = s.find("[EOS]")
        if idx >= 0:
            s = s[:idx]
        for tok in ("[BOS]", "[EOS]", "[PAD]"):
            s = s.replace(tok, "")
        return "".join(s.split())

    def compute(self, smiles_list: List[str]):
        """基于 RDKit 进行有效性判定，按 canonical SMILES 计算指标。"""
        smiles_list = smiles_list or []
        cleaned = [self._clean_smiles_text(s) for s in smiles_list]

        mols = [Chem.MolFromSmiles(s) for s in cleaned]
        valid_smiles = [Chem.MolToSmiles(m, canonical=True) for m in mols if m is not None]

        valid_count = len(valid_smiles)
        validity = valid_count / self.n_target

        unique_valid = list(set(valid_smiles))
        uniqueness = (len(unique_valid) / valid_count) if valid_count > 0 else 0.0

        diversity = float(self.evaluator(unique_valid)) if len(unique_valid) >= 2 else 0.0

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

