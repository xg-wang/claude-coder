# main function to run the demo
# 1. Reference the API key
# 2. Generate the response 

import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT, APIConnectionError, RateLimitError, APIStatusError
import gradio as gr
import httpx
from langchain import llms

MODEL_NAME = "claude-2"
MAX_TOKEN_TO_SAMPLE = 1000
MAX_RETRIES = 5

def prepare_inputs():
    inputs = 'How to get an Anthropic offer?'

    return inputs

def gen_reponse(inputs):

    api_key = os.environ.get('CLAUDE_API_KEY', None)

    client = Anthropic(api_key=api_key, timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=2.0))

    try:
        completion = client.with_options(max_retries=MAX_RETRIES).completions.create(
            prompt=f"{HUMAN_PROMPT} {inputs} {AI_PROMPT}",
            max_tokens_to_sample=MAX_TOKEN_TO_SAMPLE,
            model=MODEL_NAME,
        )
        return completion.completion
    except APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)

def main():

    # 1. Add github repo and generate the repo database via LangChain
        # Chain: 
    # 2. Accept the user input and call Anthropic API to get response

    demo = gr.Interface(fn=gen_reponse, inputs="text", outputs="text")
    
    demo.launch()   



