	reset
    python3 policy_compression_train.py \
    --log_dir "evals/log_dagger" \
    --policy_dir "evals/tmp_dagger" \
    --eval_ep_len 200 \
    --dagger_beta 35 \
    --n_iter 10 \
    --n_eval 6 \
    --seed 1

