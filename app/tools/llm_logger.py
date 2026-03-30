import logging
import tiktoken

logger = logging.getLogger("llm_usage")
logger.setLevel(logging.INFO)

# Add a console handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [LLM] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)


def _count_tokens(text: str, model: str) -> int:
    """Count tokens for a given text and model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return encoding.encode(text) if isinstance(text, str) else 0


def count_tokens(text: str, model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def log_llm_usage(model: str, caller: str, input_text: str, output_text: str):
    """Log input/output token counts for an LLM call."""
    input_tokens = count_tokens(input_text, model)
    output_tokens = count_tokens(output_text, model)
    total = input_tokens + output_tokens
    logger.info(
        f"model={model} | caller={caller} | "
        f"input_tokens={input_tokens} | output_tokens={output_tokens} | total={total}"
    )


def log_llm_stream_usage(model: str, caller: str, input_text: str, output_text: str):
    """Log token counts for a streaming LLM call (called after stream completes)."""
    log_llm_usage(model, caller, input_text, output_text)
