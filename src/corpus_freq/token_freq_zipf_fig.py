import pickle
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser(
        description="Plot token frequency vs. rank (Zipf's law) from n-gram counts"
    )
    parser.add_argument(
        "--ngram-pkl-path",
        type=Path,
        help="Path to the pickle file containing n-gram counts",
    )
    parser.add_argument(
        "--output-fig-path",
        type=Path,
        help="Path to save the output figure",
    )
    return parser.parse_args()


def main(ngram_pkl_path: Path, output_fig_path: Path, fig_size=(8, 6)):
    fig, ax = plt.subplots(figsize=fig_size)
    with open(ngram_pkl_path, "rb") as f:
        ngram_counts: list[tuple[tuple[int], int]] = pickle.load(f)

    ranks = list(range(1, len(ngram_counts) + 1))
    frequencies = [count for _, count in ngram_counts]
    ax.plot(ranks, frequencies, marker="o", linestyle="None", markersize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title("Token Frequency vs. Rank (Zipf's Law)")
    ax.grid(True, which="major", ls="--", linewidth=0.5)

    output_fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    main(
        ngram_pkl_path=args.ngram_pkl_path,
        output_fig_path=args.output_fig_path,
    )
