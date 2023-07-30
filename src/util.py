MODEL_NAME = "claude-2"
MAX_TOKEN_TO_SAMPLE = 1000
MAX_RETRIES = 5


def setup_logging(verbose = True):
    import logging
    import langchain
    if verbose:
        langchain.debug = True
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S"
    )

