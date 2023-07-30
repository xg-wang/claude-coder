# Anthropic Hackathon

## Development

```sh
git clone git@github.com:xg-wang/claude-coder.git

curl -sSL https://install.python-poetry.org | python3 -

poetry install
```

## How to run

```sh
# Run the web app
poetry run serve

# Run the claude coder agent
poetry run python src/code_agent.py --repo "https://github.com/chroma-core/chroma" --query "How do I use chromadb? Can you run an example?"

# Run the standalone code search agent
poetry run python src/code_search_tool.py --repo "https://github.com/openai/whisper" --query 'what is whisper'

# RUn the standalone code interpreter agent
poetry run python src/code_interpreter_tool.py
```


## Links

- Key: https://console.anthropic.com/account/keys
    - Store your key in .anthropic_api_key, it will be git ignored
- Anthropic Docs:
    - Get started: https://docs.anthropic.com/claude/docs
    - Prompt design: https://docs.anthropic.com/claude/docs/introduction-to-prompt-design
    - Useful hacks: https://docs.anthropic.com/claude/docs/let-claude-say-i-dont-know
    - Use-cases: https://docs.anthropic.com/claude/docs/content-generation
    - Trouble-shooting checklist: https://docs.anthropic.com/claude/docs/prompt-troubleshooting-checklist
