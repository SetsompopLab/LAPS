# MODL TRAINING

Run all from project root.

## Stage 1

**Stage 1 1D undersampling with various US levels**

```bash
python src/pips/modl/train.py \
    --device 4 \
    --project modl-slamming \
    --exp-name R1D-7-bs-8 \
    --modl_norm_type instance-affine \
    --normalize_recons_to_target \
    --unroll-iters 1 \
    --R_1d 5 7 9 \
    --retrospective_undersampling \
    --n-train-steps 10000 \
    --lr 1e-3 \
    --batch-size 8 \
    --num_workers 8
```

**Stage 1 2D undersampling with various US levels**

```bash
python src/pips/modl/train.py \
    --device 6 \
    --project modl-slamming \
    --exp-name R2D-multilevel-bs-8 \
    --modl_norm_type instance-affine \
    --normalize_recons_to_target \
    --unroll-iters 1 \
    --R_2d 20 25 30 \
    --retrospective_undersampling \
    --n-train-steps 10000 \
    --lr 1e-3 \
    --batch-size 8 \
    --num_workers 8
```

## Stage 2

**Stage 2: 1D at various US levels**

```bash
python src/pips/modl/train.py \
    --device 4 \
    --project modl-slamming \
    --exp-name R1D-multilevel-bs-1-stage2 \
    --modl_norm_type instance-affine \
    --normalize_recons_to_target \
    --scale_denoiser \
    --unroll-iters 10 \
    --R_1d 5 7 9 \
    --retrospective_undersampling \
    --n-train-steps 10000 \
    --lr 1e-4 \
    --batch-size 1 \
    --num_workers 1  \
    --checkpoint_path ./logs/R1D-7-bs-8_1unroll_slam_sim/2025-07-06_13-46/model_latest.pth
```


**Stage 2: 2D at various US levels**

```bash
python src/pips/modl/train.py \
    --device 6 \
    --project modl-slamming \
    --exp-name R2D-multilevel-bs-1-stage2 \
    --modl_norm_type instance-affine \
    --normalize_recons_to_target \
    --scale_denoiser \
    --unroll-iters 10 \
    --R_2d 20 25 30 \
    --retrospective_undersampling \
    --n-train-steps 10000 \
    --lr 1e-4 \
    --batch-size 1 \
    --num_workers 1  \
    --checkpoint_path ./logs/R2D-multilevel-bs-8_1unroll_slam_sim/2025-07-06_14-19/model_latest.pth
```


**Fine-tuned for specific rates (R=5, 10, 15, 20, 30)**
```bash
python src/pips/modl/train.py \
    --device 5 \
    --project modl-slamming \
    --exp-name R2D-R5-bs-1-stage2 \
    --modl_norm_type instance-affine \
    --normalize_recons_to_target \
    --scale_denoiser \
    --unroll-iters 10 \
    --R_2d 5 \
    --retrospective_undersampling \
    --n-train-steps 10000 \
    --lr 1e-4 \
    --batch-size 1 \
    --num_workers 1  \
    --checkpoint_path ./models/modl/model_2d_bs1_normalized_scaledenoiser_10unroll_100kstep.pth

python src/pips/modl/train.py \
    --device 6 \
    --project modl-slamming \
    --exp-name R2D-R15-bs-1-stage2 \
    --modl_norm_type instance-affine \
    --normalize_recons_to_target \
    --scale_denoiser \
    --unroll-iters 10 \
    --R_2d 15 \
    --retrospective_undersampling \
    --n-train-steps 10000 \
    --lr 1e-4 \
    --batch-size 1 \
    --num_workers 1  \
    --checkpoint_path ./models/modl/model_2d_bs1_normalized_scaledenoiser_10unroll_100kstep.pth


python src/pips/modl/train.py \
    --device 5 \
    --project modl-slamming \
    --exp-name R2D-R10-bs-1-stage2 \
    --modl_norm_type instance-affine \
    --normalize_recons_to_target \
    --scale_denoiser \
    --unroll-iters 10 \
    --R_2d 10 \
    --retrospective_undersampling \
    --n-train-steps 10000 \
    --lr 1e-4 \
    --batch-size 1 \
    --num_workers 1  \
    --checkpoint_path ./models/modl/model_2d_bs1_normalized_scaledenoiser_10unroll_100kstep.pth

python src/pips/modl/train.py \
    --device 6 \
    --project modl-slamming \
    --exp-name R2D-R20-bs-1-stage2 \
    --modl_norm_type instance-affine \
    --normalize_recons_to_target \
    --scale_denoiser \
    --unroll-iters 10 \
    --R_2d 20 \
    --retrospective_undersampling \
    --n-train-steps 10000 \
    --lr 1e-4 \
    --batch-size 1 \
    --num_workers 1  \
    --checkpoint_path ./models/modl/model_2d_bs1_normalized_scaledenoiser_10unroll_100kstep.pth

python src/pips/modl/train.py \
    --device 2 \
    --project modl-slamming \
    --exp-name R2D-R30-bs-1-stage2 \
    --modl_norm_type instance-affine \
    --normalize_recons_to_target \
    --scale_denoiser \
    --unroll-iters 10 \
    --R_2d 30 \
    --retrospective_undersampling \
    --n-train-steps 10000 \
    --lr 1e-4 \
    --batch-size 1 \
    --num_workers 1  \
    --checkpoint_path ./models/modl/model_2d_bs1_normalized_scaledenoiser_10unroll_100kstep.pth
```

**Fine-tuned for specific 1D rates (R=3, 5, 7 ,9)**
```bash
python src/pips/modl/train.py \
    --device 0 \
    --project modl-slamming \
    --exp-name R1D-R3-bs-1-stage2 \
    --modl_norm_type instance-affine \
    --normalize_recons_to_target \
    --scale_denoiser \
    --unroll-iters 10 \
    --R_1d 3 \
    --retrospective_undersampling \
    --n-train-steps 10000 \
    --lr 1e-4 \
    --batch-size 1 \
    --num_workers 1  \
    --checkpoint_path ./models/modl/model_1d_bs1_normalized_scaledenoiser_10unroll_jointR.pth

python src/pips/modl/train.py \
    --device 7 \
    --project modl-slamming \
    --exp-name R1D-R5-bs-1-stage2 \
    --modl_norm_type instance-affine \
    --normalize_recons_to_target \
    --scale_denoiser \
    --unroll-iters 10 \
    --R_1d 5 \
    --retrospective_undersampling \
    --n-train-steps 10000 \
    --lr 1e-4 \
    --batch-size 1 \
    --num_workers 1  \
    --checkpoint_path ./models/modl/model_1d_bs1_normalized_scaledenoiser_10unroll_jointR.pth

python src/pips/modl/train.py \
    --device 2 \
    --project modl-slamming \
    --exp-name R1D-R6-bs-1-stage2 \
    --modl_norm_type instance-affine \
    --normalize_recons_to_target \
    --scale_denoiser \
    --unroll-iters 10 \
    --R_1d 6 \
    --retrospective_undersampling \
    --n-train-steps 10000 \
    --lr 1e-4 \
    --batch-size 1 \
    --num_workers 1  \
    --checkpoint_path ./models/modl/model_1d_bs1_normalized_scaledenoiser_10unroll_jointR.pth

python src/pips/modl/train.py \
    --device 8 \
    --project modl-slamming \
    --exp-name R1D-R7-bs-1-stage2 \
    --modl_norm_type instance-affine \
    --normalize_recons_to_target \
    --scale_denoiser \
    --unroll-iters 10 \
    --R_1d 7 \
    --retrospective_undersampling \
    --n-train-steps 10000 \
    --lr 1e-4 \
    --batch-size 1 \
    --num_workers 1  \
    --checkpoint_path ./models/modl/model_1d_bs1_normalized_scaledenoiser_10unroll_jointR.pth

python src/pips/modl/train.py \
    --device 9 \
    --project modl-slamming \
    --exp-name R1D-R9-bs-1-stage2 \
    --modl_norm_type instance-affine \
    --normalize_recons_to_target \
    --scale_denoiser \
    --unroll-iters 10 \
    --R_1d 9 \
    --retrospective_undersampling \
    --n-train-steps 10000 \
    --lr 1e-4 \
    --batch-size 1 \
    --num_workers 1  \
    --checkpoint_path ./models/modl/model_1d_bs1_normalized_scaledenoiser_10unroll_jointR.pth
```