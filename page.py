import gradio as gr
import random
import time
from src.clone_and_embed_repo import embed_repo


def learn_repo(repo_url):
  # TODO: update reset
  if repo_url.startswith("https://github.com/"):
    embed_repo(repo_url=repo_url, reset=False)

def chat_page():
    with gr.Blocks() as demo:
        repo_url = ""
        with gr.Row():
            with gr.Column(scale=6):
                repo_url = gr.Textbox(
                    placeholder="Github Repo Link",
                    lines=1,
                    label="Github Repo Link"
                )
                print(f"repo_url: {repo_url}")
            with gr.Column(scale=2):
                learn_repo_btn = gr.Button("Learn Repo").style(full_width=True)
            with gr.Column(scale=2):
                learn_progress = gr.Textbox(label="Status")

            learn_repo_btn.click(learn_repo, repo_url, learn_progress)

        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        def respond(message, chat_history, repo_url):
            print(f"respond for message: {message}")

            db = embed_repo(repo_url=repo_url, reset=False)
            docs = db.similarity_search(message, k=3)

            bot_message_list = [
                              f"""
                              Document {i+1} \n
                              source:\n
                              {docs[i].metadata['path']}\n
                              content:\n
                              {docs[i].page_content} \n\n"""
                              for i in range(len(docs))
                              ]
            bot_message = "\n".join(bot_message_list)

            # bot_message = random.choice(["How are you?", "Good morning", "I'm very hungry"])
            print(f"bot_message is {bot_message}")
            chat_history.append((message, bot_message))
            #time.sleep(2)
            return "", chat_history

        msg.submit(respond, [msg, chatbot, repo_url], [msg, chatbot])

    demo.launch()
