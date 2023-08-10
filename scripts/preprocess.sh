#!/bin/bash

DATASET=thermal
MODEL=g2s_series_rel
TASK=reaction_prediction
REPR_START=smiles
REPR_END=smiles
N_WORKERS=8

PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/pub/opt/intel/oneapi/intelpython/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/pub/opt/intel/oneapi/intelpython/latest/etc/profile.d/conda.sh" ]; then
        . "/home/pub/opt/intel/oneapi/intelpython/latest/etc/profile.d/conda.sh"
    else
        export PATH="/home/pub/opt/intel/oneapi/intelpython/latest/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate graph2smiles


python preprocess.py \
  --model="$MODEL" \
  --data_name="$DATASET" \
  --task="$TASK" \
  --representation_start=$REPR_START \
  --representation_end=$REPR_END \
  --train_src="./data/$DATASET/src-train.txt" \
  --train_tgt="./data/$DATASET/tgt-train.txt" \
  --val_src="./data/$DATASET/src-val.txt" \
  --val_tgt="./data/$DATASET/tgt-val.txt" \
  --test_src="./data/$DATASET/src-test.txt" \
  --test_tgt="./data/$DATASET/tgt-test.txt" \
  --log_file="$PREFIX.preprocess.log" \
  --preprocess_output_path="./preprocessed/$PREFIX/" \
  --seed=42 \
  --max_src_len=1024 \
  --max_tgt_len=1024 \
  --num_workers="$N_WORKERS"
