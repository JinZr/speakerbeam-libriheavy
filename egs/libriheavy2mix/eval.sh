OMP_NUM_THREADS=1 python eval.py \
    --model_path ./small/checkpoints/epoch\=78-step\=17696.ckpt \
    --exp_dir ./small/ \
    --use_gpu 1 \
    --out_dir ./epoch78