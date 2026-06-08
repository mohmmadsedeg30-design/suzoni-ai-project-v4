import gradio as gr
import os
from chat import SuzoniChat

# إعداد محرك الدردشة
model_path = "checkpoints/suzoni_final.pt"
if not os.path.exists(model_path):
    model_path = "checkpoints/suzoni_initial.pt"

try:
    chat_engine = SuzoniChat(mp=model_path)
    personality = chat_engine.c.PERSONALITY
except Exception as e:
    print(f"Error: {e}")
    chat_engine = None

def respond(message, history):
    if chat_engine is None:
        return "Error: Model not loaded. Please run setup first."
    
    # تحويل التاريخ لتنسيق SuzoniChat إذا لزم الأمر
    # Gradio history is list of lists [[user, bot], ...]
    bot_message = chat_engine.chat(message)
    return bot_message

# تصميم الواجهة
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 🤖 {personality['name']} AI Dashboard
    ### {personality['speaking_style']} | {', '.join(personality['traits'])}
    Welcome to the interactive interface of Suzoni AI. Type your message below to start chatting!
    """)
    
    chatbot = gr.ChatInterface(
        respond,
        examples=["Hello!", "Who are you?", "Tell me a joke", "What is your logic?"],
        title=f"Chat with {personality['name']}",
    )
    
    with gr.Accordion("System Information", open=False):
        gr.Markdown(f"""
        **Model Name:** {chat_engine.c.NAME if chat_engine else 'N/A'}  
        **Version:** {chat_engine.c.VERSION if chat_engine else 'N/A'}  
        **Catchphrase:** {personality['catchphrase']}
        """)

if __name__ == "__main__":
    demo.launch(share=True)
