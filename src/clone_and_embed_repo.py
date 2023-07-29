import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

import logging
import uuid
import os
from contextlib import contextmanager
import subprocess
import pathlib
from concurrent.futures import ThreadPoolExecutor as Pool
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


def embed_repo(repo_url: str) -> Chroma:
    # Embed a repo based on its url
    repo_name = repo_url.split("/")[-1]
    collection_name = "/".join(repo_url.split("/")[-2:-1])
    repo_dir = GIT_REPOS_DIR / repo_name
    persistent_client = chromadb.PersistentClient(path=CHROMA_DB.__str__())
    # disallowed_special=() is required to avoid Exception: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte from tiktoken for some repositories
    embedding_function = OpenAIEmbeddings(disallowed_special=())

    try:
        collection = persistent_client.get_collection(
            name=collection_name, embedding_function=embedding_function.embed_documents
        )
    except ValueError:
        collection = persistent_client.create_collection(
            name=collection_name, embedding_function=embedding_function.embed_documents
        )
        executor = Pool(max_workers=5)
        with _clone_repo(repo_url, repo_dir):
            # Walk the repo directory and detect ext and load each language, ignore files that are in .gitignore
            for root, dirs, files in os.walk(repo_dir, topdown=True):
                dirs[:] = [d for d in dirs if d not in IGNORE_LIST]

                def process_one_file(file):
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

                executor.map(process_one_file, files)
    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    logging.info(
        f"Embedding repo {repo_url} has {langchain_chroma._collection.count()} documents"
    )
    return langchain_chroma


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    # openai.api_key = os.environ.get("OPENAI_API_KEY", "null")
    db = embed_repo("https://github.com/openai/whisper")
    docs = db.similarity_search("Tell me how to call whisper API in Python")
    print(len(docs))
    for doc in docs:
        print(f"File path: {doc.metadata['path']}")
        print(f"Document content:\n{doc.page_content}")
