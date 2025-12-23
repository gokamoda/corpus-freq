N=1
TOKENIZER="openai-community/gpt2"


python src/ngram_count_pkl.py \
    --n ${N} \
    --tokenizer ${TOKENIZER} \
    --save-pkl-path data/ngram_counts/openwebtext_${N}gram_${TOKENIZER}_counts.pkl \
    --corpus-dir data/corpora/openwebtext/Skylion007--openwebtext_train \
    --tokenized-corpus-dir data/tokenized_corpus/openwebtext/Skylion007--openwebtext_train \
    --num-processes 8