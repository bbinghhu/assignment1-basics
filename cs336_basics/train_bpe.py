import regex as re
from collections import defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def apply_merge(curr_counts, best_pair):
    merged_token = best_pair[0] + best_pair[1]
    new_counts = defaultdict(int)

    for tup, count in curr_counts.items():
        new_tup = []
        i = 0
        n = len(tup)

        while i < n:
            if i + 1 < n and (tup[i], tup[i + 1]) == best_pair:
                new_tup.append(merged_token)
                i += 2
            else:
                new_tup.append(tup[i])
                i += 1

        new_counts[tuple(new_tup)] += count

    return new_counts

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Initialize vocab
    vocab = {}
    new_id = 0
    for special in special_tokens:
        vocab[new_id] = special.encode("utf-8")
        new_id += 1
    for num in range(256):
        vocab[new_id] = bytes([num])
        new_id += 1
    merges = []
    

    with open(input_path, 'r', encoding="utf-8") as file:
        content = file.read()

    # Split by special tokens
    if special_tokens:
        split_pat = "|".join(re.escape(tok) for tok in special_tokens)
        parts = re.split(split_pat, content)
    else:
        parts = [content]

    # Pretokenization
    pretokens_count = defaultdict(int)
    for part in parts:
        for match in re.finditer(PAT, part):
            string_tok = match.group(0)
            byte_seq = string_tok.encode("utf-8")
            token_tuple = tuple(bytes([b]) for b in byte_seq)
            pretokens_count[token_tuple] += 1

    # Merge loop
    curr_counts = pretokens_count
    while len(vocab) < vocab_size:
        pair_counts = defaultdict(int)
        for tup, count in curr_counts.items():
            for i in range(len(tup)-1):
                pair_counts[(tup[i], tup[i+1])] += count  

        # Find the pair with the highest count
        if not pair_counts:
            break
        best_pair, _ = max(pair_counts.items(),
                                    key=lambda item : (item[1], item[0]))
        # Merge
        merges.append(best_pair)
        vocab[new_id] = best_pair[0] + best_pair[1]
        new_id += 1
        curr_counts = apply_merge(curr_counts, best_pair)
        
    return vocab, merges 