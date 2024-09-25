torchrun --standalone --nproc_per_node=2 train_gpt2.py \
    --input_bin "data/fineweb_10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb_10B/fineweb_val_*.bin" \
    --output_dir pylog768_fjlt \
    --model d12 \
    --batch_size 16 \
    --gradient_accumulation_steps 16 \
    --sequence_length 1024 \
    --val_loss_every 64 \
    --num_iterations 9536 \
    --weight_decay 0.1 \
    --learning_rate 0.0018 \
    --warmup_iters 256 \
    --warmdown_iters 2048 \
    --save_every 1000
