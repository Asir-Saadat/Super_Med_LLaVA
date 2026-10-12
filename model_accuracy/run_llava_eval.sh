#!/bin/bash
#SBATCH --job-name=llava_eval
#SBATCH --account=vlm-idk
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20g
#SBATCH --time=4:00:00
#SBATCH --output=logs/llava_eval_%j.out
#SBATCH --error=logs/llava_eval_%j.err

# Usage: sbatch run_llava_eval.sh [domain] [type]
#   domain: x_ray | mri | histology | gross_pathology | ct | all  (default: all)
#   type:   standard | upd | both                                  (default: both)

DOMAIN=${1:-all}
TYPE=${2:-both}

cd /home/as5606/projects/Super_LlaVA_Med/model_accuracy
mkdir -p logs

conda activate llava-med

python llava_eval.py \
    --domain "${DOMAIN}" \
    --type "${TYPE}" \
    --output "results_${DOMAIN}_${TYPE}.json"
