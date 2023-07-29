# main function to run the demo
# 1. Reference the API key
# 2. Generate the response 

import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT, APIConnectionError, RateLimitError, APIStatusError

MODEL_NAME = "claude-2"
MAX_TOKEN_TO_SAMPLE = 1000

def prepare_inputs():
    inputs = 'How to get an Anthropic offer?'

    return inputs



def gen_reponse():

    inputs = prepare_inputs()

    api_key = os.environ.get('CLAUDE_API_KEY', None)

    client = Anthropic(api_key=api_key)

    try:
        completion = client.completions.create(
            prompt=f"{HUMAN_PROMPT} {inputs} {AI_PROMPT}",
            max_tokens_to_sample=MAX_TOKEN_TO_SAMPLE,
            model=MODEL_NAME,
        )

        print(completion.completion)
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
    gen_reponse()
    
    


