"""
Char-level Tokenizer
====================

- Tokenizes text at character level
- No external libraries
- Deterministic & reversible
- Designed for mini language models

Author: you
"""

from typing import List, Dict


class CharTokenizer:
    def __init__(self, text: str):
        """
        Build vocabulary from raw text.

        Args:
            text (str): full training corpus
        """
        if not text:
            raise ValueError("Input text is empty")

        # get unique characters
        self.chars = sorted(list(set(text)))

        # vocabulary size
        self.vocab_size = len(self.chars)

        # mappings
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

        for idx, ch in enumerate(self.chars):
            self.stoi[ch] = idx
            self.itos[idx] = ch

    def encode(self, text: str) -> List[int]:
        """
        Convert string into list of token IDs.

        Args:
            text (str): input text

        Returns:
            List[int]: token ids
        """
        tokens = []
        for ch in text:
            if ch not in self.stoi:
                raise ValueError(f"Unknown character encountered: {repr(ch)}")
            tokens.append(self.stoi[ch])
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Convert list of token IDs back into string.

        Args:
            tokens (List[int]): token ids

        Returns:
            str: decoded text
        """
        chars = []
        for token in tokens:
            if token not in self.itos:
                raise ValueError(f"Invalid token id: {token}")
            chars.append(self.itos[token])
        return "".join(chars)

    def save_vocab(self, path: str):
        """
        Save vocabulary to file for reproducibility.
        """
        with open(path, "w", encoding="utf-8") as f:
            for ch in self.chars:
                f.write(repr(ch) + "\n")

    def __repr__(self) -> str:
        return (
            f"CharTokenizer(vocab_size={self.vocab_size})"
        )


# -------------------------------
# Simple self-test
# -------------------------------
if __name__ == "__main__":
    sample_text = "hello world\nerror occurred"
    tokenizer = CharTokenizer(sample_text)

    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print("Tokenizer:", tokenizer)
    print("Vocab size:", tokenizer.vocab_size)
    print("Encoded:", encoded[:20])
    print("Decoded:", decoded)

    assert decoded == sample_text
    print("Tokenizer self-test PASSED")
