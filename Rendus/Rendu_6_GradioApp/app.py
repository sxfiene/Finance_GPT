import keras
import gradio as gr
from huggingface_hub import hf_hub_download

link = hf_hub_download(repo_id="DiDiR6/GPT2Financial", filename="GPT2Financial_model.keras")
gpt2 = keras.models.load_model(link)

gpt2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def respond(message, chat_history):
    bot_message = gpt2.generate(message)
    return bot_message



demo = gr.ChatInterface(fn=respond, title="GPT2 Financial", retry_btn=None, undo_btn=None, clear_btn=None)
demo.launch(height=650)
