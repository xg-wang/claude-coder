MODEL_NAME = "claude-2"
MAX_TOKEN_TO_SAMPLE = 1000
MAX_RETRIES = 5


def setup_logging():
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(threadName)s] %(message)s",
    )
    logging.getLogger("langchain").setLevel(logging.INFO)
