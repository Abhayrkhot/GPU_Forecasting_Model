#!/bin/bash
# Mega Setup Script - Creates ALL files for GPU Forecasting Project
# Copy this entire file and run it on Discovery cluster

set -e

echo "=========================================="
echo "GPU Forecasting Project - Complete Setup"
echo "=========================================="

# Create directories
mkdir -p reports data models logs

echo "Creating requirements.txt..."
cat > requirements.txt << 'EOF'
torch>=2.0.0
transformers>=4.35.0
nvidia-ml-py3
pandas
numpy
scikit-learn
matplotlib
seaborn
psutil
joblib
accelerate
aiohttp
EOF

echo "Creating setup_env.sh..."
cat > setup_env.sh << 'EOF'
#!/bin/bash
echo "Setting up environment..."
module load python/3.9
module load cuda/11.8
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
echo "✓ Environment ready!"
EOF
chmod +x setup_env.sh

echo "Creating run_pipeline.slurm..."
cat > run_pipeline.slurm << 'EOF'
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

echo "GPU Forecasting Pipeline Started: $(date)"
module load python/3.9
module load cuda/11.8
source venv/bin/activate
nvidia-smi
python main_pipeline.py --model gpt2
echo "Pipeline Completed: $(date)"
EOF

echo "Creating README.md..."
cat > README.md << 'EOF'
# GPU Performance Forecasting Project

## Quick Start

1. Setup environment (first time only):
   ```bash
   ./setup_env.sh
   ```

2. Interactive run:
   ```bash
   srun --partition=gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=64GB --time=02:00:00 --pty /bin/bash
   source venv/bin/activate
   python main_pipeline.py --quick --model gpt2
   ```

3. Batch job:
   ```bash
   sbatch run_pipeline.slurm
   ```

## Files
- `benchmark_results.csv` - Performance data
- `gpu_forecaster.pkl` - Trained model
- `reports/` - Visualizations
EOF

echo "✓ All configuration files created!"
echo ""
echo "Next: Copy the Python files from the artifacts into separate .py files"
echo "Files needed:"
echo "  - main_pipeline.py"
echo "  - benchmark_gpu.py"
echo "  - forecasting_model.py"  
echo "  - visualization_report.py"
echo "  - async_load_test.py"
echo ""
echo "Then run: ./setup_env.sh"
