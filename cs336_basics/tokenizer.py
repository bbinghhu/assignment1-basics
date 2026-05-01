import pickle
from collections.abc import Iterable, Iterator

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges,
        and optional list of special tokens.
        """
        # Make copies so we do not accidentally mutate objects passed by caller.
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens is not None else []

        # Reverse vocabulary: bytes -> int.
        self.bytes_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        # Add special tokens to vocab if they are not already present.
        # Each special token should be stored as UTF-8 bytes.
        if len(self.special_tokens):
            next_id = max(self.vocab.keys()) + 1
            for tok in self.special_tokens:
                tok_bytes = tok.encode("utf-8")
                if tok_bytes not in self.bytes_to_id:
                    self.vocab[next_id] = tok_bytes
                    self.bytes_to_id[tok_bytes] = next_id
                    next_id += 1
            


        # TODO:
        # It is often useful to sort special tokens by length descending
        # before constructing a regex pattern for splitting.
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)

        # TODO:
        # Optionally construct a compiled regex pattern that matches any special token.
        # Remember to use re.escape on each special token.
        self.special_pat = None
        if len(self.special_tokens):
            self.special_pat = re.compile(
        "(" + "|".join(re.escape(tok) for tok in self.special_tokens_sorted) + ")"
            )


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Construct a Tokenizer from serialized vocab and merges files.

        The exact loading logic depends on how you serialized your vocab
        and merges in train_bpe.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
    
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)

    def _apply_merge_to_sequence(
        self,
        tokens: list[bytes],
        merge: tuple[bytes, bytes],
    ) -> list[bytes]:
        """
        Apply one BPE merge to one pre-token byte sequence.

        Example:
            tokens = [b"t", b"h", b"e"]
            merge = (b"t", b"h")
            result = [b"th", b"e"]
        """
        new = []
        i = 0
        while i < len(tokens):
            if i+1 < len(tokens) and (tokens[i], tokens[i+1]) == merge:
                new.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                new.append(tokens[i])
                i += 1
        return new


    def _encode_pretoken(self, pretoken: str) -> list[int]:
        byte_string = pretoken.encode("utf-8")
        tokens = [bytes([b]) for b in byte_string]

        for merge in self.merges:
            tokens = self._apply_merge_to_sequence(tokens, merge)

        return [self.bytes_to_id[tok] for tok in tokens]

   

    def _encode_ordinary_text(self, text: str) -> list[int]:
        """
        Encode text that does not contain special tokens.
        """
        ids: list[int] = []

        # TODO:
        # Use re.finditer(PAT, text)
        # For each regex pre-token, call self._encode_pretoken(...)
        # and extend ids.
        for match in re.finditer(PAT, text):
            pretoken = match.group(0)
            ids.extend(self._encode_pretoken(pretoken))

        return ids

    def encode(self, text: str) -> list[int]:
        """
        Encode an input string into a list of token IDs.

        Must preserve special tokens as single tokens.
        """
        # If there are no special tokens, just call _encode_ordinary_text.
        if not self.special_tokens_sorted:
            return self._encode_ordinary_text(text)

        # TODO:
        # If special tokens exist, split text into ordinary chunks and special-token chunks.
        # For ordinary chunks, call _encode_ordinary_text.
        # For special-token chunks, append the special token's ID directly.
        else:
            running_list = []
            for part in self.special_pat.split(text):
                if not part:
                    continue
                if part in self.special_tokens:
                    running_list.append(self.bytes_to_id[part.encode("utf-8")])
                else:
                    running_list.extend(self._encode_ordinary_text(part))
        return running_list
             
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings, yielding token IDs one at a time.

        A simple first version can encode each string chunk independently.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        # TODO:
        # Look up each token ID in self.vocab.
        # Concatenate the bytes.
        # Decode using UTF-8 with errors="replace".
        byte_string = b"".join(self.vocab[id] for id in ids)
        return byte_string.decode("utf-8", errors="replace")

        