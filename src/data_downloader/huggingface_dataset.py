import json
from pathlib import Path

import datasets
from datasets import IterableDataset, IterableDatasetDict
from multiprocess.queues import Queue
from tqdm import tqdm

from utils.mp_utils import PoolWithTqdm, PoolWithTqdmSingle


def save_dataset_as_jsonl(args: tuple[IterableDataset, Path, int, Queue]) -> int:
    with PoolWithTqdmSingle(args) as (_args, pos):
        dataset = _args[0]
        save_path = _args[1]
        shard_id = _args[2]
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            for instance in tqdm(
                dataset,
                desc=f"shard {shard_id}",
                position=pos,
                leave=False,
                mininterval=1,
            ):
                json.dump(instance, f, ensure_ascii=False)
                f.write("\n")
    return shard_id


def main(
    load_dataset_kwargs: dict,
    save_dir: str,
    num_processes: int = 4,
):
    dataset = datasets.load_dataset(**load_dataset_kwargs)
    assert isinstance(dataset, IterableDatasetDict)
    for split in dataset.keys():
        num_digit_shards = len(str(dataset[split].num_shards))
        shard_format = f"{load_dataset_kwargs['path'].replace('/', '--')}_{split}/{{:0{num_digit_shards}d}}.jsonl"

        args = [
            (
                dataset[split].shard(
                    num_shards=dataset[split].num_shards, index=shard_id
                ),
                Path(save_dir).joinpath(shard_format.format(shard_id)),
                shard_id,
            )
            for shard_id in range(dataset[split].num_shards)
        ]

        with PoolWithTqdm(
            num_processes=min(num_processes, len(args)), args=args
        ) as tqdm_pool:
            results = []
            for result in tqdm(
                tqdm_pool.pool.imap_unordered(
                    save_dataset_as_jsonl, tqdm_pool.args_with_queue
                ),
                total=len(tqdm_pool.args_with_queue),
                desc="Processing items",
                position=0,
                leave=False,
            ):
                results.append(result)

        print(f"done {split}:", results)
