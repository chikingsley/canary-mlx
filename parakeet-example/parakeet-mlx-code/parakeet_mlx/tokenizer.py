"""Tokenizer for ASR tasks."""


# TODO: Decode some tokens (might edit to support other variants)
def decode(tokens: list[int], vocabulary: list[str]):
    """Decode a list of tokens into a string.

    Args:
        tokens: List of integers representing tokens.
        vocabulary: List of strings representing the vocabulary.

    Returns:
        str: Decoded string.
    """
    return "".join([vocabulary[token].replace("▁", " ") for token in tokens])
