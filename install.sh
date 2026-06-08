#!/bin/bash
# Suzoni AI Auto-Installer for Ubuntu/Termux
set -e

echo "🤖 Suzoni AI Installer"
echo "======================"

# Check if running in Termux
if [ -n "$TERMUX_VERSION" ]; then
    echo "📱 Termux detected"
    pkg update -y
    pkg install -y python python-pip git
else
    echo "🖥️  Ubuntu/Debian detected"
    sudo apt update -y
    sudo apt install -y python3 python3-pip python3-venv git
fi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for compatibility)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install transformers numpy tqdm

# Create directories
mkdir -p data checkpoints logs

# Run setup
python main.py setup

echo ""
echo "✅ Installation complete!"
echo ""
echo "Usage:"
echo "  source venv/bin/activate"
echo "  python main.py chat"
echo ""
