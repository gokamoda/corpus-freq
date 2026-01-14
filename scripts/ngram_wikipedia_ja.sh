N=1
TOKENIZER="llm-jp/llm-jp-3.1-1.8b"


python src/corpus_freq/ngram_count_pkl.py \
    --n ${N} \
    --tokenizer ${TOKENIZER} \
    --save-pkl-path data/ngram_counts/wikipedia_ja_${N}gram_${TOKENIZER}_counts.pkl \
    --corpus-dir data/corpora/wikipedia/20231101_ja \
    --tokenized-corpus-dir data/tokenized_corpus/wikipedia_ja/20231101_ja \
    --num-processes 15