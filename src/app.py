import logging
import os
import gradio as gr
import os
from langchain.chat_models import ChatAnthropic
from langchain.chat_models import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from src.code_search_tool import embed_repo
from src.util import MODEL_NAME, MAX_RETRIES, MAX_TOKEN_TO_SAMPLE, setup_logging
from src.code_agent import initialize_claude_coder, run_agent_qa
from termcolor import colored




def gen_retriever(repo_url, reset):
    db = embed_repo(repo_url, reset)
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20

    return retriever

def gen_response(message, chat_history, repo_url):
    llm = ChatAnthropic(model=MODEL_NAME,
                        anthropic_api_key=os.environ.get('CLAUDE_API_KEY', None),
                        max_tokens_to_sample=MAX_TOKEN_TO_SAMPLE)

    retriever = gen_retriever(repo_url, reset=False)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
    ## TODO: update chat_history
    result = qa({"question": message, "chat_history": []})
    bot_message = result["answer"]
    chat_history.append((message, bot_message))
    return "", chat_history

def learn_repo(repo_url):
  # TODO: update reset
    if repo_url.startswith("https://github.com/"):
        embed_repo(repo_url=repo_url, reset=False)
        global agent
        agent = initialize_claude_coder(repo_url)
        return "OK"
    else:
        print("Repo URL must starts with https")
        return "ERROR"


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


        with gr.Row():

            # with gr.Column(scale=5):
            #     chatbot = gr.Chatbot().style(height=750)
            #     msg = gr.Textbox()
            #     clear = gr.ClearButton([msg, chatbot])
            #     msg.submit(gen_response, [msg, chatbot, repo_url], [msg, chatbot])
            with gr.Column(scale=5):
                logbot = gr.Chatbot().style(height=750)
                message = gr.Textbox()
                
                message.submit(user, [message, logbot], [message, logbot], queue=True).then(gen_log, [message, logbot], logbot)
    demo.queue()
    demo.launch()


def user(user_message, history):
    return user_message, history + [[ user_message, None]]

def gen_log(query, history):
    global agent
    history[-1][1] = ""
    for step in run_agent_qa(query, agent):
        logging.info(f"Step: {step}")
        if output := step.get("intermediate_step"):
            action, value = output[0]
            # history[-1][1] += f"action:\n{action.tool}"
            history[-1][1] += f"Tool input:\n{action.tool_input} \n"
            history[-1][1] += f"Action:\n{action.tool} \n"
            history[-1][1] += f"Value:\n{value} \n"
            # logging.info(f"action:\n{action.tool}")
            # logging.info(f"value:\n{value}")
        if output:=step.get("output"):
            history[-1][1] += f"Output: {output}"
        yield history
        # else:
        #     output = step.get("output")
        #     logging.info(f"Output: {output}")
        #     history[-1][1] += f"Output: {output}"
        #     yield history

def main():
    setup_logging(True)
    # 1.1 Add github repo and generate the repo database via LangChain
        # Repo clone
        # Text splitter
        # Chroma persistence
    # 1.2 Context generation based on user input
        # Similarity search in chromdb given user input
        # re-generate the prompt: user input + context
    # 2. Anthropic API to get final response
    chat_page()
