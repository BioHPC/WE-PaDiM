#!/usr/bin/env bash
# Setup script for WE-PaDiM

set -e

echo "==============================================="
echo "WE-PaDiM Setup Script"
echo "==============================================="

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $PYTHON_VERSION"

# Check if running in virtual environment
if [[ -z "${VIRTUAL_ENV}" ]] && [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo ""
    echo "WARNING: Not running in a virtual environment!"
    echo "It is recommended to create a virtual environment first:"
    echo "  conda create -n wepadim python=3.10"
    echo "  conda activate wepadim"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."
echo "If you need a different CUDA version, edit this script or install manually."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import pytorch_wavelets; print('pytorch_wavelets: OK')"

echo ""
echo "==============================================="
echo "Setup complete!"
echo "==============================================="
echo ""
echo "Next steps:"
echo "1. Download datasets to data/ directory (see data/README.md)"
echo "2. Run a quick test:"
echo "   PYTHONPATH=./src python scripts/run_experiment.py --help"
echo ""
echo "For detailed usage, see README.md"
echo "==============================================="
