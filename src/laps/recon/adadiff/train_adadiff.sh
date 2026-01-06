accelerate launch src/pips/recon/adadiff/train_adadiff_multicoil.py \
    --dataset slam \
    --exp adadiff_slam \
    --num_epoch 500 \
    --batch_size 12 \
    --wandb_project adadiff \
    --wandb_run_suffix SLAM \
    --log_freq 20 \
    --save_content_every 10 \
    --save_content
