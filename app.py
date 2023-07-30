import os
import gradio as gr
from util import MODEL_NAME, MAX_RETRIES, MAX_TOKEN_TO_SAMPLE
import os
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatAnthropic
from app import MODEL_NAME
from src.clone_and_embed_repo import embed_repo
from langchain.chat_models import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
import page

def gen_retriever(repo_url):
    db = embed_repo(repo_url, reset=False)
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20

    return retriever

def get_repo_url():
    # TODO: get from gr text input
    return "https://github.com/openai/whisper"

def gen_response(inputs):
    llm = ChatAnthropic(model=MODEL_NAME, anthropic_api_key=os.environ.get('CLAUDE_API_KEY', None))
    retriever = gen_retriever(get_repo_url())
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
    result = qa({"question": inputs, "chat_history": []})
    return result["answer"]

def main():

    # 1.1 Add github repo and generate the repo database via LangChain
        # Repo clone
        # Text splitter
        # Chroma persistence
    # 1.2 Context generation based on user input
        # Similarity search in chromdb given user input
        # re-generate the prompt: user input + context
    # 2. Anthropic API to get final response

    # page.chat_page()

    demo = gr.Interface(fn=gen_response, inputs="text", outputs="text")

    demo.launch()




