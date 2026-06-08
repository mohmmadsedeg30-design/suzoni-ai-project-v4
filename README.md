# Suzoni AI 🤖

Custom AI with built-in reasoning and enhanced logic layers. Supports both Arabic and English.

## 🌟 New: Interactive Interfaces
We have added beautiful and easy-to-use interfaces for everyone!

### 1. Terminal UI (Colorful & Professional)
Experience Suzoni in your terminal with colors and panels.
```bash
python tui.py
```

### 2. Web Dashboard (User Friendly)
Launch a web-based chat interface that you can even share with friends!
```bash
python web_ui.py
```

---

## 🚀 Quick Start

1. **Setup:**
   ```bash
   python main.py setup
   ```
2. **Train (Optional):**
   ```bash
   python main.py train --data your_data.txt
   ```
3. **Chat (Basic):**
   ```bash
   python main.py chat
   ```

## 🛠️ Advanced Integration
Integrate with external models like GPT-2:
```bash
python main.py integrate --model gpt2
```

## 📂 Project Structure
- `tui.py` - **New** Colorful Terminal Interface
- `web_ui.py` - **New** Web-based Interactive Dashboard
- `config.py` - Model & System Settings
- `model.py` - Neural Network Architecture (with Logic Layers)
- `tokenizer.py` - Custom BPE Tokenizer
- `trainer.py` - Training Logic
- `chat.py` - Core Chat Engine
- `main.py` - Command Line Entry Point
