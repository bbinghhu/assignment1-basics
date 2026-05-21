import regex as re
import os
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def merge_tuple(tup, best_pair):
    """
    Merge every non-overlapping occurrence of best_pair inside tup.
    Example:
        tup = (b'a', b'b', b'a', b'b')
        best_pair = (b'a', b'b')
        returns (b'ab', b'ab')
    """
    merged_token = best_pair[0] + best_pair[1]
    new_tup = []

    i = 0
    while i < len(tup):
        if i + 1 < len(tup) and (tup[i], tup[i + 1]) == best_pair:
            new_tup.append(merged_token)
            i += 2
        else:
            new_tup.append(tup[i])
            i += 1

    return tuple(new_tup)


def add_word_pairs(word_id, tup, count, pair_counts, pair_to_word_ids):
    """
    Add all adjacent-pair contributions from one pre-token tuple.
    """
    for i in range(len(tup) - 1):
        pair = (tup[i], tup[i + 1])
        pair_counts[pair] += count
        pair_to_word_ids[pair].add(word_id)


def remove_word_pairs(word_id, tup, count, pair_counts, pair_to_word_ids):
    """
    Remove all adjacent-pair contributions from one pre-token tuple.
    """
    seen_pairs = set()

    for i in range(len(tup) - 1):
        pair = (tup[i], tup[i + 1])

        pair_counts[pair] -= count
        if pair_counts[pair] == 0:
            del pair_counts[pair]

        # Only update pair_to_word_ids once per distinct pair in this word.
        if pair not in seen_pairs:
            pair_to_word_ids[pair].discard(word_id)
            if not pair_to_word_ids[pair]:
                del pair_to_word_ids[pair]
            seen_pairs.add(pair)


def contains_pair(tup, pair):
    """
    Check whether pair occurs as an adjacent pair in tup.
    """
    for i in range(len(tup) - 1):
        if (tup[i], tup[i + 1]) == pair:
            return True
    return False


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

  
    # Initialize vocabulary
    vocab = {}
    new_id = 0

    for special in special_tokens:
        vocab[new_id] = special.encode("utf-8")
        new_id += 1

    for num in range(256):
        vocab[new_id] = bytes([num])
        new_id += 1

    merges = []

    
    # Read input file
    pretokens_count = defaultdict(int)

    with open(input_path, "rb") as f:
        if special_tokens:
            # For TinyStories this is usually b"<|endoftext|>"
            split_special_token = special_tokens[0].encode("utf-8")
            boundaries = find_chunk_boundaries(
                f,
                desired_num_chunks=4,
                split_special_token=split_special_token,
            )
        else:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            boundaries = [0, file_size]

        split_pat = None
        if special_tokens:
            split_pat = "|".join(re.escape(tok) for tok in special_tokens)

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            if split_pat is not None:
                parts = re.split(split_pat, chunk)
            else:
                parts = [chunk]

            for part in parts:
                for match in re.finditer(PAT, part):
                    string_tok = match.group(0)
                    byte_seq = string_tok.encode("utf-8")
                    token_tuple = tuple(bytes([b]) for b in byte_seq)
                    pretokens_count[token_tuple] += 1
    # Instead of repeatedly storing a dict from tuple -> count,
    # give each distinct pre-token an integer ID.
    word_tokens = list(pretokens_count.keys())
    word_counts = list(pretokens_count.values())

    pair_counts = defaultdict(int)
    pair_to_word_ids = defaultdict(set)

    for word_id, tup in enumerate(word_tokens):
        count = word_counts[word_id]
        add_word_pairs(word_id, tup, count, pair_counts, pair_to_word_ids)

    
    # Merge loop
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # Pick highest-frequency pair.
        # Tie-break by lexicographically larger pair.
        best_pair, best_count = max(
            pair_counts.items(),
            key=lambda item: (item[1], item[0])
        )

        merges.append(best_pair)

        merged_token = best_pair[0] + best_pair[1]
        vocab[new_id] = merged_token
        new_id += 1

        # Only these pre-tokens can change.
        affected_word_ids = list(pair_to_word_ids[best_pair])

        for word_id in affected_word_ids:
            old_tup = word_tokens[word_id]
            count = word_counts[word_id]

            # Safety check: because structures are updated during the loop.
            if not contains_pair(old_tup, best_pair):
                continue

            # Remove old pair contributions.
            remove_word_pairs(
                word_id,
                old_tup,
                count,
                pair_counts,
                pair_to_word_ids,
            )

            # Merge this pre-token.
            new_tup = merge_tuple(old_tup, best_pair)
            word_tokens[word_id] = new_tup

            # Add new pair contributions.
            add_word_pairs(
                word_id,
                new_tup,
                count,
                pair_counts,
                pair_to_word_ids,
            )

    return vocab, merges