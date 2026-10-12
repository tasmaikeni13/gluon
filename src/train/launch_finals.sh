#!/bin/bash
# Finals: 4 arms, 40k steps each. One run per GPU at all times (clean timing).
# GPU0 lane: LM AdamW -> Vision AdamW ; GPU1 lane: LM GLUON -> Vision GLUON.
# Vision lrs filled in by the launcher (ADAMW_VLR / GLUON_VLR env).
cd /home/navin/problem/gluon
LANE=$1
if [ "$LANE" = "0" ]; then
  export CUDA_VISIBLE_DEVICES=0
  python3 src/train/train.py --domain lm --optimizer adamw --lr 1e-3 --steps 40000 \
    --run-name final_lm_adamw 2>&1 | tail -2
  python3 src/train/train.py --domain vision --optimizer adamw --lr ${ADAMW_VLR} --steps 40000 \
    --run-name final_vit_adamw 2>&1 | tail -2
  echo LANE0_DONE
else
  export CUDA_VISIBLE_DEVICES=1
  python3 src/train/train.py --domain lm --optimizer gluon --lr 4e-4 --gamma 0.02 --steps 40000 \
    --run-name final_lm_gluon 2>&1 | tail -2
  python3 src/train/train.py --domain vision --optimizer gluon --lr ${GLUON_VLR} --gamma 0.02 --steps 40000 \
    --run-name final_vit_gluon 2>&1 | tail -2
  echo LANE1_DONE
fi
