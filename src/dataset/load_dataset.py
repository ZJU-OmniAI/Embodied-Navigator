import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from datasets import Dataset as HFDataset, concatenate_datasets

from ..utils import dataset_registry


# ---------------- 基础包装 ----------------

class HFDatasetWrapper(Dataset):
    """将 🤗 Dataset 适配为 PyTorch Dataset。"""

    def __init__(self, ds: HFDataset):
        self.ds = ds

    def __len__(self):
        return self.ds.num_rows

    def __getitem__(self, i):
        return self.ds[i]


@dataclass
class BuiltOne:
    ds: Dataset
    length: int
    name: str
    weight: float
    start: Optional[int]
    num_samples: Optional[int]


def _concat_many_torch(datasets: List[Dataset]) -> Dataset:
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _concat_many_hf(datasets: List[HFDataset]) -> HFDataset:
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


# ---------------- 加载与标准化 ----------------

def _build_one_cfg(cfg: Dict[str, Any]) -> BuiltOne:
    ds_type = cfg["type"]
    base_path = cfg["path"]
    splits = list(cfg["splits"])
    weight = float(cfg.get("weight", 1.0))
    start = cfg.get("start", 0)
    num_samples = cfg.get("num_samples", None)

    parts_hf, parts_torch = [], []
    for sp in splits:
        part = dataset_registry.create(ds_type, base_path=base_path, split=sp)
        if isinstance(part, HFDataset):
            parts_hf.append(part)
        else:
            parts_torch.append(part)

    if parts_hf and parts_torch:
        logging.warning(
            f"[{ds_type}] 同一数据集的不同 split 返回类型不一致，"
            f"将把 🤗 Dataset 包装为 PyTorch Dataset 后再拼接。"
        )
        parts_torch.extend(HFDatasetWrapper(p) for p in parts_hf)
        merged = _concat_many_torch(parts_torch)
    elif parts_hf:
        merged_hf = _concat_many_hf(parts_hf)
        merged = HFDatasetWrapper(merged_hf)
    else:
        merged = _concat_many_torch(parts_torch)

    length = len(merged)

    logging.info(
        f"Loaded dataset: type={ds_type}, "
        f"path={base_path}, split={splits}, size={length}"
    )

    return BuiltOne(ds=merged, length=length, name=ds_type, weight=weight, start=start, num_samples=num_samples)


def _build_all(cfgs: List[Dict[str, Any]]) -> List[BuiltOne]:
    return [_build_one_cfg(c) for c in cfgs]


# ---------------- 交错计划 ----------------

def _plan_concat(built: List[BuiltOne]) -> Dataset:
    return _concat_many_torch([b.ds for b in built])


class PlannedDataset(Dataset):
    """基于预计算 plan 的多数据集交错访问。"""

    def __init__(self, datasets: Sequence[Dataset], plan: List[tuple[int, int]]):
        self.datasets = list(datasets)
        self.plan = plan

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, idx):
        while True:
            src_id, offset = self.plan[idx]
            data = self.datasets[src_id][offset]
            return data
            try:
                src_id, offset = self.plan[idx]
                data = self.datasets[src_id][offset]
                return data
            except:
                print(
                    f"[error] while getting data from {idx}({src_id}-{offset})", end="")
                idx = idx + 1 if idx + 1 >= self.__len__() else 0
                print(f", change to {idx}")


def _proportional_seq_plan(built: List[BuiltOne]) -> List[tuple[int, int]]:
    total_len = sum(b.length for b in built)
    weights = [max(0.0, b.weight) for b in built]
    if sum(weights) == 0:
        weights = [1.0] * len(weights)
    sum_w = sum(weights)
    cur_w = [0.0] * len(built)
    remain = [b.length for b in built]
    offsets = [0] * len(built)
    plan: List[tuple[int, int]] = []

    for _ in range(total_len):
        for i in range(len(built)):
            if remain[i] > 0:
                cur_w[i] += weights[i]
            else:
                cur_w[i] = -1e18
        j = max(range(len(built)), key=lambda k: cur_w[k])
        if remain[j] == 0:
            found = False
            for k in range(len(built)):
                if remain[k] > 0:
                    j = k
                    found = True
                    break
            if not found:
                break
        plan.append((j, offsets[j]))
        offsets[j] += 1
        remain[j] -= 1
        cur_w[j] -= sum_w

    return plan


def _proportional_rand_plan(built: List[BuiltOne], seed: int) -> List[tuple[int, int]]:
    rnd = random.Random(seed)
    total_len = sum(b.length for b in built)
    weights = [max(0.0, b.weight) for b in built]
    if sum(weights) == 0:
        weights = [1.0] * len(weights)
    sum_w = sum(weights)

    remain = [b.length for b in built]
    offsets = [0] * len(built)
    plan: List[tuple[int, int]] = []

    for _ in range(total_len):
        r = rnd.random() * sum_w
        acc, pick = 0.0, None
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                pick = i
                break
        if pick is None:
            pick = len(built) - 1

        tries = 0
        while remain[pick] == 0 and tries < len(built):
            pick = (pick + 1) % len(built)
            tries += 1
        if remain[pick] == 0:
            break

        plan.append((pick, offsets[pick]))
        offsets[pick] += 1
        remain[pick] -= 1

    return plan


def _count_seq_plan(built: List[BuiltOne]) -> List[tuple[int, int]]:
    targets = []
    for b in built:
        if b.num_samples is None:
            logging.warning(f"[{b.name}] 未提供 num_samples，按全部样本使用：{b.length}")
            t = b.length
        else:
            t = int(b.num_samples)
            if t > b.length:
                logging.warning(
                    f"[{b.name}] 目标数 {t} 超过可用样本 {b.length}，将截断为 {b.length}")
                t = b.length
        targets.append(t)

    remain = targets[:]
    offsets = [0] * len(built)
    plan: List[tuple[int, int]] = []

    while sum(remain) > 0:
        progressed = False
        for i in range(len(built)):
            if remain[i] > 0:
                plan.append((i, offsets[i]))
                offsets[i] += 1
                remain[i] -= 1
                progressed = True
        if not progressed:
            break
    return plan


def _count_rand_plan(built: List[BuiltOne], seed: int) -> List[tuple[int, int]]:
    rng = np.random.default_rng(seed)

    pairs = []
    for i, b in enumerate(built):
        if b.num_samples is None:
            t = b.length
        else:
            t = min(int(b.num_samples + b.start), b.length)
        pairs.extend((i, j) for j in range(b.start, t))

    pairs = np.array(pairs, dtype=np.int64)
    rng.shuffle(pairs)

    return [tuple(p) for p in pairs]

# ---------------- 主函数 ----------------


def load_and_merge_datasets(
    dataset_cfgs: List[Dict[str, Any]],
    mode: str = "concat",
    seed: int = 42,
) -> Dataset:
    """
    mode:
        - "concat": 直接合并
        - "prop_seq": 按权重顺序交错
        - "prop_rand": 按权重随机交错
        - "count_seq": 按数目顺序交错
        - "count_rand": 按数目随机交错
    """
    if mode == "no_warp":
        cfg = dataset_cfgs[0]
        dtype = cfg.pop("type")
        base_path = cfg.pop("path")
        split = list(cfg.pop("splits"))[0]
        return dataset_registry.create(dtype, base_path=base_path, split=split, **cfg)

    built = _build_all(dataset_cfgs)

    if mode == "concat":
        return _plan_concat(built)

    datasets = [b.ds for b in built]

    if mode == "prop_seq":
        plan = _proportional_seq_plan(built)
        return PlannedDataset(datasets, plan)
    elif mode == "prop_rand":
        plan = _proportional_rand_plan(built, seed=seed)
        return PlannedDataset(datasets, plan)
    elif mode == "count_seq":
        plan = _count_seq_plan(built)
        return PlannedDataset(datasets, plan)
    elif mode == "count_rand":
        plan = _count_rand_plan(built, seed=seed)
        return PlannedDataset(datasets, plan)
    else:
        raise ValueError(f"未知合并模式: {mode}")


@dataclass
class DatasetArguments:
    datasets: Optional[List[Dict[str, Any]]] = field(
        default_factory=list,
        metadata={"help": "多个数据集配置，每个包含 name 和 splits 列表"},
    )
    dataset_merge_type: Optional[str] = field(
        default="concat",
        metadata={"help": "多个数据集合并配置"},
    )


def build_hf_datasets(pt_ds, sample_size: int = 1000) -> HFDataset:
    sample_size = min(sample_size, len(pt_ds))
    sample_indices = random.sample(range(len(pt_ds)), sample_size)
    samples = [pt_ds[i] for i in sample_indices]

    sample_dict = {
        "messages": [s["messages"] for s in samples],
        "length": [int(s.get("length", 0)) for s in samples],
    }
    sample_hf = HFDataset.from_dict(sample_dict)
    features = sample_hf.features

    def gen():
        for i in range(len(pt_ds)):
            item = pt_ds[i]
            yield {
                "messages": item["messages"],
                "length": int(item["length"]),
            }

    hf_ds = HFDataset.from_generator(gen, features=features)
    return hf_ds
