

OMP_NUM_THREADS=1 python eval.py \
    --model_path ./medium/checkpoints/epoch\=78-step\=17696.ckpt \
    --exp_dir ./medium/ \
    --use_gpu 1 \
    --out_dir ./epoch78