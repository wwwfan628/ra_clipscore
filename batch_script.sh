#!/bin/bash
#SBATCH -J train_coop
#SBATCH --gpus 1
#SBATCH -t 0-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yifanlu@kth.se
#SBATCH --output /proj/berzelius-2024-331/users/x_yiflu/interpretable_evaluation_metrics/logs/%J_train_coop.out
#SBATCH --error  /proj/berzelius-2024-331/users/x_yiflu/interpretable_evaluation_metrics/logs/%J_train_coop.err

cd /proj/berzelius-2024-331/users/x_yiflu/interpretable_evaluation_metrics

# Execute your code
/proj/berzelius-2024-331/users/x_yiflu/.conda/envs/py310/bin/python train_eval_coop.py --exp 'celebA' --dataset_dir /proj/berzelius-2024-331/users/x_yiflu/dataset/Datasets/face1k_furniture1k --n_attr 3 --attr_type 'USER' --num_epochs 100