import gradio as gr
import random
import time

def learn_repo(repo_url):
    return random.choice(["OK", "?"])

def chat_page():
    with gr.Blocks() as demo:

        with gr.Row():
            with gr.Column(scale=6):
                repo_url = gr.Textbox(
                    placeholder="Github Repo Link",
                    lines=1,
                    label="Github Repo Link"
                )
            with gr.Column(scale=2):
                learn_repo_btn = gr.Button("Learn Repo").style(full_width=True)
            with gr.Column(scale=2):
                learn_progress = gr.Textbox(label="Status")

            learn_repo_btn.click(learn_repo, repo_url, learn_progress)

        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        def respond(message, chat_history):
            bot_message = random.choice(["How are you?", "Good morning", "I'm very hungry"])
            chat_history.append((message, bot_message))
            #time.sleep(2)
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])

    demo.launch()
