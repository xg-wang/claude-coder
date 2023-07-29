# main function to run the demo
# 1. Reference the API key
# 2. Generate the response 

import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

MODEL_NAME = "claude-2"
MAX_TOKEN_TO_SAMPLE = 1000

def gen_reponse(inputs):
    pass

def main():
    api_key = os.environ.get('CLAUDE_API_KEY', None)
    print(api_key)
    anthropic = Anthropic(
        api_key=api_key,
    )

    completion = anthropic.completions.create(
        model=MODEL_NAME,
        max_tokens_to_sample=MAX_TOKEN_TO_SAMPLE,
        prompt=f"{HUMAN_PROMPT} how does a court case get to the Supreme Court? {AI_PROMPT}",
    )
    print(completion.completion)


