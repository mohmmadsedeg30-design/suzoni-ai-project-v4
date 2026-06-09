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
2. **Train (Optional, but Recommended for Smarter AI):**
   To make Suzoni AI truly intelligent, train it with the provided dataset:
   ```bash
   python main.py train --data training_data.txt
   ```
   You can also create your own `your_data.txt` and train with it.
   
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
- `training_data.txt` - **New** Sample dataset for training the model.
- `app.py` - **New** Gradio application for Hugging Face Spaces deployment.
- `tui.py` - Colorful Terminal Interface.
- `web_ui.py` - Web-based Interactive Dashboard.
- `config.py` - Model & System Settings.
- `model.py` - Neural Network Architecture (with Logic Layers).
- `tokenizer.py` - Custom BPE Tokenizer.
- `trainer.py` - Training Logic.
- `chat.py` - Core Chat Engine.
- `main.py` - Command Line Entry Point.

## 🚀 Deploy to Hugging Face Spaces

To make your Suzoni AI publicly available, you can deploy it to Hugging Face Spaces:

1.  **Create a new Space:** Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space.
2.  **Choose Gradio SDK:** Select `Gradio` as the SDK.
3.  **Upload Files:** Upload all files from your `suzoni-ai-project-v4` directory to your Hugging Face Space, ensuring `app.py` is in the root.
4.  **Install Dependencies:** In your Space settings, add `torch`, `rich`, `gradio` to `requirements.txt`.
5.  **Run Setup:** Ensure you run `python main.py setup` in your Space's terminal (or locally and upload the `checkpoints` and `tokenizer.json` files) to initialize the model and tokenizer.

Your Suzoni AI will then be live and accessible to everyone!
