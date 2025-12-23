N=1
TOKENIZER="openai-community/gpt2"


python src/ngram_count_pkl.py \
    --n ${N} \
    --tokenizer ${TOKENIZER} \
    --save-pkl-path data/ngram_counts/tinystories_${N}gram_${TOKENIZER}_counts.pkl \
    --corpus-dir data/corpora/tinystories/roneneldan--TinyStories_train \
    --tokenized-corpus-dir data/tokenized_corpus/tinystories/roneneldan--TinyStories_train \
    --num-processes 8