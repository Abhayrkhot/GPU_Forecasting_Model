#!/bin/bash
#SBATCH --job-name=gpu_forecast
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --output=logs/gpu_forecast_%j.out
#SBATCH --error=logs/gpu_forecast_%j.err

echo "=========================================="
echo "GPU Performance Forecasting Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

module load python/3.9
module load cuda/11.8

source venv/bin/activate

echo ""
echo "GPU Information:"
nvidia-smi
echo ""

python main_pipeline.py --model gpt2

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Pipeline completed successfully!"
else
    echo ""
    echo "✗ Pipeline failed!"
    exit 1
fi

echo "End Time: $(date)"

