import gradio as gr
import os
from chat import SuzoniChat

# إعداد محرك الدردشة
# Hugging Face Spaces عادة ما تكون في بيئة Linux، لذا المسارات ستكون مباشرة
model_path = "checkpoints/suzoni_final.pt"
tokenizer_path = "tokenizer.json"

# التأكد من وجود الملفات، وإلا نستخدم الإعداد الأولي
if not os.path.exists(model_path):
    print(f"Warning: {model_path} not found. Attempting to use initial model.")
    model_path = "checkpoints/suzoni_initial.pt"
    if not os.path.exists(model_path):
        print("Error: Initial model not found. Please run 'python main.py setup' first.")
        # يمكننا هنا استدعاء setup إذا أردنا، لكن يفضل أن يكون النموذج جاهزاً
        # from main import setup
        # setup()
        # model_path = "checkpoints/suzoni_initial.pt"

if not os.path.exists(tokenizer_path):
    print("Error: Tokenizer not found. Please run 'python main.py setup' first.")
    # يمكننا هنا استدعاء setup إذا أردنا

try:
    chat_engine = SuzoniChat(mp=model_path, tp=tokenizer_path)
    personality = chat_engine.c.PERSONALITY
except Exception as e:
    print(f"Error loading Suzoni AI for Gradio: {e}")
    chat_engine = None
    # تعيين قيم افتراضية في حالة فشل التحميل
    class DefaultConfig:
        NAME = "Suzoni AI"
        VERSION = "1.0"
        PERSONALITY = {
            "name": "Suzoni",
            "speaking_style": "Helpful",
            "traits": ["Intelligent", "Friendly"],
            "catchphrase": "I am here to assist"
        }
    chat_engine = type('obj', (object,), {'c': DefaultConfig()})()
    personality = chat_engine.c.PERSONALITY

def respond(message, history):
    if chat_engine is None:
        return "Error: Model not loaded. Please ensure setup is complete."
    
    # Gradio history is list of lists [[user, bot], ...]
    # SuzoniChat expects a list of dicts [{'u': user, 's': bot}, ...]
    # We need to convert Gradio history to SuzoniChat history format
    chat_engine.h = [] # Clear previous history for each new session in Gradio
    for human, ai in history:
        if human and ai:
            chat_engine.h.append({"u": human, "s": ai})

    bot_message = chat_engine.chat(message)
    return bot_message

# تصميم الواجهة
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 🤖 {personality["name"]} AI Dashboard
    ### {personality["speaking_style"]} | {", ".join(personality["traits"])}
    Welcome to the interactive interface of Suzoni AI. Type your message below to start chatting!
    """)
    
    chatbot = gr.ChatInterface(
        respond,
        examples=["Hello!", "Who are you?", "Tell me a joke", "What is your logic?"],
        title=f"Chat with {personality["name"]}",
    )
    
    with gr.Accordion("System Information", open=False):
        gr.Markdown(f"""
        **Model Name:** {chat_engine.c.NAME}  
        **Version:** {chat_engine.c.VERSION}  
        **Catchphrase:** {personality["catchphrase"]}
        """)

if __name__ == "__main__":
    demo.launch()
