#!/bin/bash
# Plot experiment results from W&B.
# Run from the KestRL root directory.

ENV="HalfCheetah-v5"
PBSAC_GROUP="sac_$ENV"

uv run python src/plots/plot_wandb_experiments.py \
    --groups "$PBSAC_GROUP" \
    --metric "pac_bayes/kl_div" \
    --smooth 0.2 \
    --font-size-add 12.0 \
    --raw-opacity 0.08 \
    --figsize 12 8 \
    --save "kl_div.pdf" \
    --environment "$ENV" \
    --hide-xlabel
