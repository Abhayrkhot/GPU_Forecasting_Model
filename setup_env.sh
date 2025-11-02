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

