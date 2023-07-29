# main function to run the demo
# 1. Reference the API key
# 2. Generate the response 

import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

MODEL_NAME = "claude-2"
MAX_TOKEN_TO_SAMPLE = 1000

def prepare_inputs():
    inputs = 'How to get an Anthropic offer?'

    return inputs



def gen_reponse(inputs):

    inputs = prepare_inputs()

    api_key = os.environ.get('CLAUDE_API_KEY', None)

    anthropic = Anthropic(
        api_key=api_key,
    )

    completion = anthropic.completions.create(
        model=MODEL_NAME,
        max_tokens_to_sample=MAX_TOKEN_TO_SAMPLE,
        prompt=f"{HUMAN_PROMPT} {inputs} {AI_PROMPT}",
    )

    return completion.completion

def main():
    print(gen_reponse)
    
    


