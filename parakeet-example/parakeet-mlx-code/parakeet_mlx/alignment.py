"""Utilities for aligning and merging token sequences from audio transcription."""

from dataclasses import dataclass


@dataclass
class AlignedToken:
    """Represents a single token with timing information."""

    id: int
    text: str
    start: float
    duration: float
    end: float = 0.0  # temporary

    def __post_init__(self) -> None:
        """Calculate the end time after initialization."""
        self.end = self.start + self.duration


@dataclass
class AlignedSentence:
    """Represents a sentence composed of aligned tokens."""

    text: str
    tokens: list[AlignedToken]
    start: float = 0.0  # temporary
    end: float = 0.0  # temporary
    duration: float = 0.0  # temporary

    def __post_init__(self) -> None:
        """Calculate sentence timings based on its tokens."""
        self.tokens = sorted(self.tokens, key=lambda x: x.start)
        self.start = self.tokens[0].start
        self.end = self.tokens[-1].end
        self.duration = self.end - self.start


@dataclass
class AlignedResult:
    """Represents the final alignment result, containing text and sentences."""

    text: str
    sentences: list[AlignedSentence]

    def __post_init__(self) -> None:
        """Strip whitespace from the final text."""
        self.text = self.text.strip()

    @property
    def tokens(self) -> list[AlignedToken]:
        """Return a flat list of all tokens from all sentences."""
        return [token for sentence in self.sentences for token in sentence.tokens]


def tokens_to_sentences(tokens: list[AlignedToken]) -> list[AlignedSentence]:
    """
    Convert a list of tokens into a list of sentences.

    Sentences are split based on punctuation marks. This is a basic
    implementation and may not cover all edge cases perfectly.

    Args:
        tokens: A list of AlignedToken objects.

    Returns:
        A list of AlignedSentence objects.
    """
    sentences = []
    current_tokens = []

    for idx, token in enumerate(tokens):
        current_tokens.append(token)

        # hacky, will fix
        if (
            "!" in token.text
            or "?" in token.text
            or "。" in token.text
            or "？" in token.text  # noqa: RUF001
            or "！" in token.text  # noqa: RUF001
            or (
                "." in token.text
                and (idx == len(tokens) - 1 or " " in tokens[idx + 1].text)
            )
        ):
            sentence_text = "".join(t.text for t in current_tokens)
            sentence = AlignedSentence(text=sentence_text, tokens=current_tokens)
            sentences.append(sentence)

            current_tokens = []

    if current_tokens:
        sentence_text = "".join(t.text for t in current_tokens)
        sentence = AlignedSentence(text=sentence_text, tokens=current_tokens)
        sentences.append(sentence)

    return sentences


def sentences_to_result(sentences: list[AlignedSentence]) -> AlignedResult:
    """
    Convert a list of sentences into a single result object.

    Args:
        sentences: A list of AlignedSentence objects.

    Returns:
        An AlignedResult object.
    """
    return AlignedResult("".join(sentence.text for sentence in sentences), sentences)


def merge_longest_contiguous(
    a: list[AlignedToken],
    b: list[AlignedToken],
    *,
    overlap_duration: float,
) -> list[AlignedToken]:
    """
    Merge two overlapping token sequences based on the longest contiguous match.

    This function identifies the longest stretch of identical token IDs where the
    start times are within a specified `overlap_duration`. It then uses this
    contiguous block to stitch the two sequences together.

    Args:
        a: The first list of tokens.
        b: The second list of tokens.
        overlap_duration: The maximum allowed time difference for tokens to be
          considered overlapping.

    Returns:
        A merged list of tokens.

    Raises:
        RuntimeError: If a sufficiently long contiguous sequence cannot be found.
    """
    if not a or not b:
        return a if a else b

    a_end_time = a[-1].end
    b_start_time = b[0].start

    if a_end_time <= b_start_time:
        return a + b

    overlap_a = [token for token in a if token.end > b_start_time - overlap_duration]
    overlap_b = [token for token in b if token.start < a_end_time + overlap_duration]

    if len(overlap_a) < 2 or len(overlap_b) < 2:
        cutoff_time = (a_end_time + b_start_time) / 2
        return [t for t in a if t.end <= cutoff_time] + [
            t for t in b if t.start >= cutoff_time
        ]

    enough_pairs = len(overlap_a) // 2

    best_contiguous: list[tuple[int, int]] = []
    for i, token_a in enumerate(overlap_a):
        for j, token_b in enumerate(overlap_b):
            if (
                token_a.id == token_b.id
                and abs(token_a.start - token_b.start) < overlap_duration / 2
            ):
                current: list[tuple[int, int]] = []
                k, m = i, j
                while (
                    k < len(overlap_a)
                    and m < len(overlap_b)
                    and overlap_a[k].id == overlap_b[m].id
                    and abs(overlap_a[k].start - overlap_b[m].start)
                    < overlap_duration / 2
                ):
                    current.append((k, m))
                    k += 1
                    m += 1

                if len(current) > len(best_contiguous):
                    best_contiguous = current

    if len(best_contiguous) >= enough_pairs:
        a_start_idx = len(a) - len(overlap_a)
        lcs_indices_a = [a_start_idx + pair[0] for pair in best_contiguous]
        lcs_indices_b = [pair[1] for pair in best_contiguous]

        result: list[AlignedToken] = []
        result.extend(a[: lcs_indices_a[0]])

        for i in range(len(best_contiguous)):
            idx_a = lcs_indices_a[i]
            idx_b = lcs_indices_b[i]

            result.append(a[idx_a])

            if i < len(best_contiguous) - 1:
                next_idx_a = lcs_indices_a[i + 1]
                next_idx_b = lcs_indices_b[i + 1]

                gap_tokens_a = a[idx_a + 1 : next_idx_a]
                gap_tokens_b = b[idx_b + 1 : next_idx_b]

                if len(gap_tokens_b) > len(gap_tokens_a):
                    result.extend(gap_tokens_b)
                else:
                    result.extend(gap_tokens_a)

        result.extend(b[lcs_indices_b[-1] + 1 :])
        return result
    else:
        raise RuntimeError(f"No pairs exceeding {enough_pairs}")


def merge_longest_common_subsequence(
    a: list[AlignedToken],
    b: list[AlignedToken],
    *,
    overlap_duration: float,
) -> list[AlignedToken]:
    """
    Merge two overlapping token sequences using the longest common subsequence.

    This function finds the longest common subsequence of token IDs within the
    overlapping region and uses it to merge the two sequences. This can be more
    robust than `merge_longest_contiguous` if there are minor discrepancies in
    the overlapping region.

    Args:
        a: The first list of tokens.
        b: The second list of tokens.
        overlap_duration: The maximum allowed time difference for tokens to be
          considered overlapping.

    Returns:
        A merged list of tokens.
    """
    if not a or not b:
        return a if a else b

    a_end_time = a[-1].end
    b_start_time = b[0].start

    if a_end_time <= b_start_time:
        return a + b

    overlap_a = [token for token in a if token.end > b_start_time - overlap_duration]
    overlap_b = [token for token in b if token.start < a_end_time + overlap_duration]

    if len(overlap_a) < 2 or len(overlap_b) < 2:
        cutoff_time = (a_end_time + b_start_time) / 2
        return [t for t in a if t.end <= cutoff_time] + [
            t for t in b if t.start >= cutoff_time
        ]

    dp = [[0 for _ in range(len(overlap_b) + 1)] for _ in range(len(overlap_a) + 1)]

    for i, token_a in enumerate(overlap_a):
        for j, token_b in enumerate(overlap_b):
            if (
                token_a.id == token_b.id
                and abs(token_a.start - token_b.start) < overlap_duration / 2
            ):
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    lcs_pairs: list[tuple[int, int]] = []
    i, j = len(overlap_a), len(overlap_b)

    while i > 0 and j > 0:
        if (
            overlap_a[i - 1].id == overlap_b[j - 1].id
            and abs(overlap_a[i - 1].start - overlap_b[j - 1].start)
            < overlap_duration / 2
        ):
            lcs_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs_pairs.reverse()

    if not lcs_pairs:
        cutoff_time = (a_end_time + b_start_time) / 2
        return [t for t in a if t.end <= cutoff_time] + [
            t for t in b if t.start >= cutoff_time
        ]

    a_start_idx = len(a) - len(overlap_a)
    lcs_indices_a = [a_start_idx + pair[0] for pair in lcs_pairs]
    lcs_indices_b = [pair[1] for pair in lcs_pairs]

    result: list[AlignedToken] = []

    result.extend(a[: lcs_indices_a[0]])

    for i in range(len(lcs_pairs)):
        idx_a = lcs_indices_a[i]
        idx_b = lcs_indices_b[i]

        result.append(a[idx_a])

        if i < len(lcs_pairs) - 1:
            next_idx_a = lcs_indices_a[i + 1]
            next_idx_b = lcs_indices_b[i + 1]

            gap_tokens_a = a[idx_a + 1 : next_idx_a]
            gap_tokens_b = b[idx_b + 1 : next_idx_b]

            if len(gap_tokens_b) > len(gap_tokens_a):
                result.extend(gap_tokens_b)
            else:
                result.extend(gap_tokens_a)

    result.extend(b[lcs_indices_b[-1] + 1 :])

    return result
