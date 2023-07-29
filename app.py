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

def gen_response(inputs):
    api_key = os.environ.get('CLAUDE_API_KEY', None)
    
    llm = ChatAnthropic(model=MODEL_NAME, anthropic_api_key=api_key)

    repo = "https://github.com/openai/whisper"
    db = embed_repo(repo, reset=False)
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20

    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

    chat_history = []

    result = qa({"question": inputs, "chat_history": chat_history})
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




