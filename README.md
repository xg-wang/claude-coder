# Anthropic Hackathon

## Development

```sh
git clone git@github.com:xg-wang/claude-coder.git

mkdir scratch # you can experiment with anything here, it's git ignored

echo '<anthropic-api-key>' > .anthropic_api_key

python3.10 -m venv .venv

source .venv/bin/activate

curl -sSL https://install.python-poetry.org | python3 -

poetry install
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
