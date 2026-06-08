#!/bin/bash
# Quick launcher for Suzoni

cd "$(dirname "$0")"
source venv/bin/activate

echo "🤖 Suzoni AI Launcher"
echo "===================="
echo "1. Train model"
echo "2. Chat with Suzoni"
echo "3. Integrate external model"
echo "4. Generate text"
echo "5. Exit"
read -p "Select: " choice

case $choice in
    1) read -p "Data file: " data; python main.py train --data "$data" ;;
    2) python main.py chat ;;
    3) read -p "Model name (gpt2/gpt2-medium/bloom-560m): " model; python main.py integrate --model "$model" ;;
    4) read -p "Prompt: " prompt; python main.py generate --prompt "$prompt" ;;
    5) exit 0 ;;
    *) echo "Invalid choice" ;;
esac
