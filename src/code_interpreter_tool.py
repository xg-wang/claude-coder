from typing import Optional
from io import BytesIO
import logging
import docker
import subprocess
from docker.errors import BuildError, ContainerError, ImageNotFound, APIError
from docker import DockerClient
from pydantic import Field
from langchain.tools.base import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from dotenv import load_dotenv, find_dotenv
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
import langchain

def clean_code(code: str) -> str:
    code = code.strip()
    code = code.removeprefix("```python\n")
    code = code.removeprefix("`\n")
    code = code.removesuffix("```")
    code = code.removesuffix("`")
    return code

class CodeInterpreterTool(BaseTool):
    name = "code_interpreter"
    description = "Executes Python code in a Docker container. It returns the log output of the Python code."
    client: DockerClient = Field(default_factory=docker.from_env)

    DOCKERFILE = """
    FROM python:3.10
    RUN pip install numpy pandas matplotlib seaborn pydantic chromadb
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = docker.from_env()
        logging.info("Building Docker image")
        try:
            self.client.images.build(
                fileobj=BytesIO(self.DOCKERFILE.encode('utf-8')), tag="python_code_interpreter:latest"
            )
        except BuildError as ex:
            logging.exception(f"Docker image build failed: {ex.msg}")
            for line in ex.build_log:
                logging.info(line)

    def _run(
        self,
        code: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run code in a Docker container."""
        code = clean_code(code)
        logging.info(f"Running code\n```\n{code}\n```\n")
        logging.info(f"run_manager: {run_manager}")
        # Use subprocess.run to call docker run, and return the stdio or stderr
        p = subprocess.run(
            ["docker", "run", "--rm", "-i", "python_code_interpreter:latest"],
            input=code.encode('utf-8'),
            capture_output=True,
            check=False,
        )
        if p.returncode != 0:
            logging.info(f"returncode: {p.returncode}")
            logging.info(f"stdout: {p.stdout}")
            logging.info(f"stderr: {p.stderr}")
            return str(p.stderr, encoding='utf-8') or "The code is not valid"
        else:
            return str(p.stdout, encoding='utf-8')

    async def _arun(
        self,
        code: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


if __name__ == "__main__":
    langchain.debug = False

    load_dotenv(find_dotenv())
    logging.basicConfig(level=logging.INFO)

    tool = CodeInterpreterTool()
    agent = initialize_agent(
        [tool],
        ChatAnthropic(temperature=0, model='claude-2'),
        # ChatOpenAI(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    agent.run("<Instruction>When you have the answer, always say 'Final Answer:'</Instruction><Instruction>You should only write valid Python program</Instruction>\thow to use ChromaDB functionalities? can you give an example")
