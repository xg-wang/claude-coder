import click
from dotenv import load_dotenv, find_dotenv
from src.util import setup_logging
from src.clone_and_embed_repo import embed_repo
from src.code_interpreter_tool import CodeInterpreterTool

def initialize_claude_coder():
    tools = [CodeInterpreterTool()]


# poetry run python src/clone_and_embed_repo.py --reset --repo "https://github.com/openai/whisper" --query 'what is whisper'
# Use python click to create a CLI, --reset to reset the database
@click.command()
@click.option('--repo', required=True, help='The URL of the repo.')
@click.option('--query', required=True, help='The query for LLM.')
@click.option('--reset', is_flag=True, default=False, help='Reset ChromaDB.')
def main(repo, query, reset):
    setup_logging()
    load_dotenv(find_dotenv())
    db = embed_repo(repo, reset=reset)
    docs = db.similarity_search(query, k=10)
    print(f"Documents length: {len(docs)}")
    for doc in docs:
        print(f"\nFile path: {doc.metadata['path']}")
        print(f"Document content:\n{doc.page_content}")


if __name__ == "__main__":
    main()