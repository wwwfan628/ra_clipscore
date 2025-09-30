#!/bin/bash
#SBATCH -J eval
#SBATCH --gpus 1
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yifanlu@kth.se
#SBATCH --output /proj/berzelius-2024-331/users/x_yiflu/interpretable_evaluation_metrics/logs/%J_imagenet.out
#SBATCH --error  /proj/berzelius-2024-331/users/x_yiflu/interpretable_evaluation_metrics/logs/%J_imagenet.err

cd /proj/berzelius-2024-331/users/x_yiflu/interpretable_evaluation_metrics

# Execute your code
# /proj/berzelius-2024-331/users/x_yiflu/.conda/envs/py310/bin/python train_eval_coop.py --exp 'celebA' --dataset_dir /proj/berzelius-2024-331/users/x_yiflu/dataset/Datasets/face1k_furniture1k --n_attr 3 --attr_type 'USER' --num_epochs 200
# /proj/berzelius-2024-331/users/x_yiflu/.conda/envs/py310/bin/python main_20250417.py --exp celebA --real_dir /proj/berzelius-2024-331/users/x_yiflu/dataset/Datasets/ffhq/ffhq_5000 --gen_dir /proj/berzelius-2024-331/users/x_yiflu/stylegan2-ada-pytorch/out --n_attr 40 --attr_type USER --score_type dualCS
/proj/berzelius-2024-331/users/x_yiflu/.conda/envs/py310/bin/python main_try_local_vit_imagenet.py --exp imagenet_exp --real_dir /proj/berzelius-2024-331/users/x_yiflu/dataset/Datasets/imagenet256 --gen_dir /proj/berzelius-2024-331/users/x_yiflu/dataset/Datasets/Imagenet256-LDM