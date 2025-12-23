import json
import pickle
from argparse import ArgumentParser
from collections import Counter
from functools import partial
from pathlib import Path

from multiprocess.queues import Queue
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.logger import init_logging
from utils.mp_utils import PoolWithTqdm, PoolWithTqdmSingle

logger = init_logging(__name__)


def parse_args():
    parser = ArgumentParser(
        description="Count n-grams in a corpus and save as a pickle file"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Size of n-grams to count",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Name or path of the tokenizer to use",
    )
    parser.add_argument(
        "--save-pkl-path",
        type=Path,
        help="Path to save the n-gram counts",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        help="Directory containing the corpus JSON files",
    )
    parser.add_argument(
        "--tokenized-corpus-dir",
        type=Path,
        help="Directory to save/load the tokenized corpus JSON files",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of processes to use for counting n-grams",
    )

    return parser.parse_args()


def count_single_process(
    n: int,
    tokenizer_name_or_path: str,
    tokenized_corpus_dir: Path,
    args_with_queue: tuple[Path, Queue[int]],
    truncate: bool = False,
) -> Counter:
    with PoolWithTqdmSingle(args_with_queue) as (args, pos):
        corpus_path = args[0]
        tokenized_corpus_path = tokenized_corpus_dir.joinpath(
            corpus_path.stem + "_tokenized.jsonl"
        )
        tokenizing_corpus_path = tokenized_corpus_dir.joinpath(
            corpus_path.stem + "_tokenizing.jsonl"
        )

        if not tokenized_corpus_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

            if truncate:
                max_length = tokenizer.model_max_length
                if max_length > 10000000:
                    truncation_args = {
                        "truncation": False,
                    }
                else:
                    logger.warn_once(
                        f"Truncating input sequences to {max_length} tokens"
                    )
                    truncation_args = {
                        "truncation": True,
                        "max_length": max_length,
                    }
            else:
                truncation_args = {
                    "truncation": False,
                    "verbose": False,
                }

            # with (
            #     open(corpus_path, encoding="utf-8") as f,
            #     open(tokenizing_corpus_path, "w", encoding="utf-8") as out_f,
            # ):
            #     for line in tqdm(
            #         f,
            #         desc=f"Tokenizing {corpus_path.name}",
            #         position=pos,
            #         leave=False,
            #         mininterval=1,
            #     ):
            #         text = json.loads(line)["text"]
            #         tokenized_line = tokenizer(text, **truncation_args)["input_ids"]
            #         out_f.write(json.dumps({"text": tokenized_line}) + "\n")
            batch_size = 4096  # tune this

            with (
                open(corpus_path, encoding="utf-8") as f,
                open(tokenizing_corpus_path, "w", encoding="utf-8") as out_f,
            ):
                batch_texts = []

                for line in tqdm(
                    f,
                    desc=f"Tokenizing {corpus_path.name}",
                    position=pos,
                    leave=False,
                    mininterval=1,
                ):
                    batch_texts.append(json.loads(line)["text"])

                    if len(batch_texts) >= batch_size:
                        enc = tokenizer(batch_texts, **truncation_args)
                        for ids in enc["input_ids"]:
                            out_f.write(json.dumps({"text": ids}) + "\n")
                        batch_texts.clear()

                # flush remainder
                if batch_texts:
                    enc = tokenizer(batch_texts, **truncation_args)
                    for ids in enc["input_ids"]:
                        out_f.write(json.dumps({"text": ids}) + "\n")
            tokenizing_corpus_path.rename(tokenized_corpus_path)

        counter = Counter()

        num_lines = sum(1 for _ in open(tokenized_corpus_path, encoding="utf-8"))
        with open(tokenized_corpus_path, encoding="utf-8") as f:
            for line in tqdm(
                f,
                desc=f"Counting {n}-grams in {corpus_path.name}",
                position=pos,
                total=num_lines,
                mininterval=1,
                leave=False,
            ):
                tokenized_line = json.loads(line)["text"]
                if len(tokenized_line) < n:
                    continue
                counter.update(zip(*[tokenized_line[i:] for i in range(n)]))

    return counter


def count(
    ngram_size: int,
    tokenizer_name_or_path: str,
    corpus_dir: Path,
    tokenized_corpus_dir: Path,
    num_processes: int = 4,
) -> Counter[tuple[int]]:
    """Count n-grams in a corpus using multiple processes.

    Parameters
    ----------
    ngram_size : int
    corpus_name : str
    tokenizer_name_or_path : str
    corpus_dir : str
        Path to the corpus directory containing JSON files.
    num_processes : int, optional
        Number of processes to use, by default 4

    Returns
    -------
    Counter
        A Counter object containing the n-gram counts

    Raises
    ------
    FileNotFoundError
        If the corpus directory does not exist
    NotADirectoryError
        If the corpus directory is not a directory
    """

    # get all json files in index_dir including subdirectories recursively
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory {corpus_dir} does not exist.")
    if not corpus_dir.is_dir():
        raise NotADirectoryError(f"Corpus directory {corpus_dir} is not a directory.")

    json_files = list(corpus_dir.glob("**/*.json*"))
    json_files.sort()  # Sort files to ensure consistent order across runs

    # prepare for counting
    combined_counter = Counter()
    count_single_process_curry = partial(
        count_single_process,
        ngram_size,
        tokenizer_name_or_path,
        tokenized_corpus_dir,
    )
    args = [(file_path,) for file_path in json_files]

    # count n-grams in parallel
    with (
        PoolWithTqdm(
            num_processes=min(num_processes, len(json_files)),
            args=args,
        ) as tqdm_pool,
        logger.timer(f"Processing n-grams with {num_processes} processes"),
    ):
        for idx, result in tqdm(
            enumerate(
                tqdm_pool.pool.imap_unordered(
                    count_single_process_curry,
                    tqdm_pool.args_with_queue,
                )
            ),
            total=len(json_files),
            desc="Counting n-grams",
        ):
            logger.debug("merging result %d/%d", idx + 1, len(json_files))
            if idx == 0:
                combined_counter = result
            else:
                combined_counter.update(result)
            logger.debug("Done merging %d/%d", idx + 1, len(json_files))

    return combined_counter


def write_counter_to_file_single_process(
    tokenizer_name_or_path: str,
    args_with_queue: tuple[Counter, Path, Queue[int]],
):
    with PoolWithTqdmSingle(args_with_queue) as (args, pos):
        counter, save_path = args
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        with open(
            save_path,
            "w",
            encoding="utf-8",
        ) as f:
            for _i, (k, v) in tqdm(
                enumerate(
                    counter,
                ),
                desc=f"Saving to {save_path.name}",
                position=pos,
                total=len(counter),
                mininterval=1,
                leave=False,
            ):
                f.write(
                    json.dumps(
                        {
                            "token_ids": list(k),
                            "text": tokenizer.decode(list(k)),
                            "count": v,
                        }
                    )
                    + "\n"
                )


def main(
    ngram_size: int,
    tokenizer_name_or_path: str,
    save_pkl_path: Path,
    corpus_dir: Path,
    tokenized_corpus_dir: Path,
    num_processes: int = 4,
):
    counts: Counter = count(
        ngram_size=ngram_size,
        tokenizer_name_or_path=tokenizer_name_or_path,
        corpus_dir=corpus_dir,
        tokenized_corpus_dir=tokenized_corpus_dir,
        num_processes=num_processes,
    )

    if ngram_size == 1:
        logger.info(
            f"Number of tokens in the corpus: {sum(counts.values()):,}",
        )

    with logger.timer("Sorting n-gram counts"):
        counts: list[tuple[tuple[int], int]] = counts.most_common()

    with (
        open(save_pkl_path, "wb") as f,
        logger.timer(f"Saving n-gram counts to {save_pkl_path}"),
    ):
        pickle.dump(counts, f)


if __name__ == "__main__":
    args = parse_args()

    args.save_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    args.tokenized_corpus_dir.mkdir(parents=True, exist_ok=True)

    main(
        ngram_size=args.n,
        tokenizer_name_or_path=args.tokenizer,
        save_pkl_path=args.save_pkl_path,
        corpus_dir=args.corpus_dir,
        tokenized_corpus_dir=args.tokenized_corpus_dir,
        num_processes=args.num_processes,
    )
