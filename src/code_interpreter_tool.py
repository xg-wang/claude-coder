from typing import Optional
from io import BytesIO
import logging
import docker
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
from langchain.llms import Anthropic
from langchain.chat_models import ChatAnthropic
import langchain

class CodeInterpreterTool(BaseTool):
    name = "code_interpreter"
    description = "Executes Python code in a Docker container. It returns the log output of the Python code."
    client: DockerClient = Field(default_factory=docker.from_env)

    DOCKERFILE = """
    FROM python:3.10
    RUN pip install numpy pandas matplotlib seaborn pydantic
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
        code = code.strip()
        if code.startswith("```python"):
            code = code.removeprefix("```python\n")
            code = code.removesuffix("```")
        logging.info(f"Running code\n```\n{code}\n```\n")
        logging.info(f"run_manager: {run_manager}")
        try:
            logs = self.client.containers.run(
                "python_code_interpreter:latest",
                command=["python", "-c", code],
                detach=False,
                stdout=True,
                stderr=True,
                auto_remove=True,
                mem_limit='1g',
                network_disabled=False,
            )
            return str(logs, encoding='utf-8')
        except ContainerError as ex:
            logging.exception(f"Container error: {ex}")
            return str(ex.stderr, encoding='utf-8')
        except ImageNotFound as ex:
            logging.exception(f"Image not found: {ex}")
            return str(ex.stderr, encoding='utf-8')
        except APIError as ex:
            logging.exception(f"API error: {ex}")
            return str(ex.stderr, encoding='utf-8')

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
        # Anthropic(temperature=0, model='claude-2'),
        # ChatOpenAI(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    agent.run("<Instruction>When you have the answer, always say 'Final Answer:'</Instruction>\tTell me how to run a basic example with pydantic")
