import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

import click
import logging
import uuid
import os
from contextlib import contextmanager
import subprocess
import pathlib
from dotenv import load_dotenv, find_dotenv


GIT_REPOS_DIR = pathlib.Path(__file__).parent.parent / "git_repos"
CHROMA_DB = pathlib.Path(__file__).parent.parent / ".chroma_db"

IGNORE_LIST = [".git", "node_modules", "__pycache__", ".idea", ".vscode"]

logging.basicConfig(level=logging.INFO)


def _ext_to_lang(ext: str) -> Language:
    # Convert a file extension to a language
    ext = ext.removeprefix(".")
    if ext == "py":
        return Language.PYTHON
    elif ext == "rs":
        return Language.RUST
    elif ext == "rb":
        return Language.RUBY
    elif ext == "md":
        return Language.MARKDOWN
    elif ext in ("ts", "tsx", "jsx", "js"):
        return Language.JS
    else:
        for lang in Language:
            if lang.value == ext:
                return lang
    raise ValueError(f"File extension {ext} not supported")


@contextmanager
def _clone_repo(repo_url: str, repo_dir: pathlib.Path):
    GIT_REPOS_DIR.mkdir(exist_ok=True)
    # Create a context manager, that clones a repo based on its url then cleans up
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, repo_dir], cwd=GIT_REPOS_DIR
        )
    try:
        yield
    finally:
        subprocess.run(["rm", "-rf", repo_dir])


def embed_repo(repo_url: str, reset: bool = False) -> Chroma:
    # Embed a repo based on its url
    repo_name = repo_url.split("/")[-1]
    collection_name = "_".join(repo_url.split("/")[-2:])
    repo_dir = GIT_REPOS_DIR / repo_name
    persistent_client = chromadb.PersistentClient(path=CHROMA_DB.__str__())
    # disallowed_special=() is required to avoid Exception: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte from tiktoken for some repositories
    embedding_function = OpenAIEmbeddings(disallowed_special=())
    try:
        if reset:
            logging.info(f"Resetting database collection {collection_name}")
            persistent_client.delete_collection(collection_name)
        else:
            logging.info(f"Retrieving database collection {collection_name}")
            collection = persistent_client.get_collection(
                name=collection_name, embedding_function=embedding_function.embed_documents
            )
    except ValueError:
        logging.info(f"Creating database collection {collection_name}")
        collection = persistent_client.create_collection(
            name=collection_name, embedding_function=embedding_function.embed_documents
        )
        with _clone_repo(repo_url, repo_dir):
            for root, dirs, files in os.walk(repo_dir, topdown=True):
                dirs[:] = [d for d in dirs if d not in IGNORE_LIST]

                for file in files:
                    logging.info(f"Processing {file}")
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file_path)[1]
                    try:
                        lang = _ext_to_lang(file_ext)
                    except ValueError:
                        logging.info(f"File extension {file_ext} not supported")
                        continue
                    with open(file_path, "r") as f:
                        code = f.read()
                        logging.info(f"Embedding {file_path} as {lang.value}")
                        lang_splitter = RecursiveCharacterTextSplitter.from_language(
                            language=lang, chunk_size=1024, chunk_overlap=0
                        )
                        lang_docs = lang_splitter.create_documents(
                            texts=[code],
                            metadatas=[{"language": lang.value, "path": file_path}],
                        )
                        for doc in lang_docs:
                            collection.add(
                                ids=[str(uuid.uuid4())],
                                documents=[doc.page_content],
                                metadatas=[doc.metadata],
                            )

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    logging.info(
        f"Embedding repo {repo_url} has {langchain_chroma._collection.count()} documents"
    )
    return langchain_chroma


# poetry run python src/clone_and_embed_repo.py --reset --repo "https://github.com/openai/whisper" --query 'what is whisper'
# Use python click to create a CLI, --reset to reset the database
@click.command()
@click.option('--repo', required=True, help='The URL of the repo.')
@click.option('--query', required=True, help='The query for LLM.')
@click.option('--reset', is_flag=True, default=False, help='Reset ChromaDB.')
def main(repo, query, reset):
    load_dotenv(find_dotenv())
    db = embed_repo(repo, reset=reset)
    docs = db.similarity_search(query, k=10)
    print(f"Documents length: {len(docs)}")
    for doc in docs:
        print(f"\nFile path: {doc.metadata['path']}")
        print(f"Document content:\n{doc.page_content}")


if __name__ == "__main__":
    main()