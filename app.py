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

def gen_response(inputs):

    load_dotenv(find_dotenv())
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


    # api_key = os.environ.get('CLAUDE_API_KEY', None)

    # client = Anthropic(api_key=api_key)

    # try:
    #     completion = client.with_options(max_retries=MAX_RETRIES).completions.create(
    #         prompt=f"{HUMAN_PROMPT} {inputs} {AI_PROMPT}",
    #         max_tokens_to_sample=MAX_TOKEN_TO_SAMPLE,
    #         model=MODEL_NAME,
    #     )
    #     return completion.completion
    # except APIConnectionError as e:
    #     print("The server could not be reached")
    #     print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    # except RateLimitError as e:
    #     print("A 429 status code was received; we should back off a bit.")
    # except APIStatusError as e:
    #     print("Another non-200-range status code was received")
    #     print(e.status_code)
    #     print(e.response)





def main():

    # 1.1 Add github repo and generate the repo database via LangChain
        # Repo clone
        # Text splitter
        # Chroma persistence
    # 1.2 Context generation based on user input
        # Similarity search in chromdb given user input
        # re-generate the prompt: user input + context
    # 2. Anthropic API to get final response
        # 

    page.chat_page()

    demo = gr.Interface(fn=gen_response, inputs="text", outputs="text")

    demo.launch()




