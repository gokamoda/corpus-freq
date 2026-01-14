from argparse import ArgumentParser
from pathlib import Path

from data_downloader.huggingface_dataset import main as hf_main


def main(
    dataset_name: str,
    save_dir: str,
    num_processes: int = 4,
):
    if dataset_name in [
        "DrNicefellow/fineweb-edu-sample-1BT",
        "Skylion007/openwebtext",
        "roneneldan/TinyStories",
        "wikimedia/wikipedia/20231101.ja",
    ]:
        if "wikimedia/wikipedia" in dataset_name:
            load_dataset_kwargs = {
                "path": "wikimedia/wikipedia",
                "name": "20231101.ja",
                "streaming": True,
            }
        else:
            load_dataset_kwargs = {
                "path": dataset_name,
                "streaming": True,
            }
        hf_main(load_dataset_kwargs, save_dir, num_processes)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Download dataset and save as jsonl files with multiprocessing"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="DrNicefellow/fineweb-edu-sample-1BT",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Example: data/fineweb-edu-1bt",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of processes to use for downloading and saving shards",
    )
    args = parser.parse_args()

    main(
        dataset_name=args.dataset,
        save_dir=args.save_dir,
        num_processes=args.num_processes,
    )
