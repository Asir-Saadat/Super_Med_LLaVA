#!/bin/bash
#SBATCH --job-name=clipup_train
#SBATCH --account=vlm-idk
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20g
#SBATCH --time=36:00:00
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

# Usage:
#   sbatch train_domain.sh <domain> <run_name>
#
# domain options: x_ray | mri | histology | gross_pathology | ct | combined
# Example:
#   sbatch train_domain.sh combined dataset3_combined_run1

DOMAIN=${1:-x_ray}
RUN_NAME=${2:-${DOMAIN}_run1}

case $DOMAIN in
    x_ray)         DATA_ROOT="/home/as5606/projects/Super_LlaVA_Med/dataset_3/x_ray" ;;
    mri)           DATA_ROOT="/home/as5606/projects/Super_LlaVA_Med/dataset_3/MRI" ;;
    histology)     DATA_ROOT="/home/as5606/projects/Super_LlaVA_Med/dataset_3/histology" ;;
    gross_pathology) DATA_ROOT="/home/as5606/projects/Super_LlaVA_Med/dataset_3/gross_pathology" ;;
    ct)            DATA_ROOT="/home/as5606/projects/Super_LlaVA_Med/dataset_3/CT" ;;
    combined)         DATA_ROOT="/home/as5606/projects/Super_LlaVA_Med/dataset_3/combined" ;;
    matched_combined) DATA_ROOT="/home/as5606/projects/Super_LlaVA_Med/dataset_3_matched/combined" ;;
    *)
        echo "Unknown domain: $DOMAIN"
        echo "Choose from: x_ray, mri, histology, gross_pathology, ct, combined, matched_combined"
        exit 1 ;;
esac

echo "Domain    : $DOMAIN"
echo "Data root : $DATA_ROOT"
echo "Run name  : $RUN_NAME"

mkdir -p logs

cd /home/as5606/projects/CLIP-UP

conda run -n clip_up python train.py \
    --vlm med-llava \
    --injection_type clip_up_emb \
    --vqa_type open_ended \
    --data_root ${DATA_ROOT} \
    --output_root_dir /home/as5606/projects/Super_LlaVA_Med/model_checkpoints \
    --run_name ${RUN_NAME}
