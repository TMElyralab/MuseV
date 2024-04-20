from __future__ import annotations

from typing import Union, Iterable, Iterator, List
import random

import pandas as pd
from torch.utils.data.sampler import BatchSampler, Sampler

from .PreextractH5pyDataset import PreextractH5pyDataset


class NumVisCondBatchSampler(BatchSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self,
        dataset: PreextractH5pyDataset,
        batch_size: int,
        drop_last: bool,
        sampler: Sampler[int] | Iterable[int] = None,
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.dataset = dataset
        self.generate_batch_lst()

    def _generate_single_batch_sample(self, df):
        idx_batch_lst = []
        batch = []
        for cnt, idx in enumerate(df["index"].sample(frac=1.0)):
            batch.append(idx)
            if (cnt + 1) % self.batch_size == 0:
                idx_batch_lst.append(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            idx_batch_lst.append(batch)
        return idx_batch_lst

    def generate_batch_lst(self):
        # TODO:// 重新生成索引，会导致一个 batch 的 n_vision_cond 不一致，
        # 怀疑是 dataloader 中的 dataset 和初始定义的 dataset 发生了变化
        # 现阶段需要在每个 dataloader 迭代器用完后或者迭代前，重新初始化一遍：如
        # dataloader.dataset.prepare_init_datas()
        #   for batch in dataloader:

        # self.dataset.prepare_init_datas()
        n_vision_cond_lst = self.dataset.n_vision_cond_lst
        mv_tail_latent2viscond_flag = self.dataset.mv_tail_latent2viscond_flag
        for i, flag in enumerate(mv_tail_latent2viscond_flag):
            if flag:
                n_vision_cond_lst[i] += 1
        n_vision_cond = zip(*[range(len(n_vision_cond_lst)), n_vision_cond_lst])
        df = pd.DataFrame(n_vision_cond, columns=["index", "n_vision_cond"])
        df_batch_group = df.groupby("n_vision_cond").apply(
            self._generate_single_batch_sample
        )
        df_batch_group = df_batch_group.reset_index()[0]
        df_batch_group = df_batch_group.tolist()

        batch_lst = []
        for i, df_batch in enumerate(df_batch_group):
            batch_lst.extend(df_batch)
        random.shuffle(batch_lst)
        self.batch_lst = batch_lst

    def __iter__(self) -> Iterator[List[int]]:
        self.generate_batch_lst()
        for batch in self.batch_lst:
            yield batch

    def __len__(self) -> int:
        return len(self.batch_lst)
