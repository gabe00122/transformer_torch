from sentiment_lm_torch.constants import (
    END_TOKEN,
    SPECIAL_TOKENS,
    EMPTY_TOKEN,
    STAR_TOKENS,
    START_TOKEN,
)
import sentencepiece as spm


class Tokenizer:
    def __init__(self, vocab_path: str, context_size: int):
        self.vocab = spm.SentencePieceProcessor(model_file=vocab_path)
        self.context_size = context_size

    def encode(self, text: str, stars: int | None = None) -> tuple[list[int], int]:
        vocab_tokens = self.vocab.Encode(text, out_type=int)

        tokens = [START_TOKEN]
        tokens.extend([t + SPECIAL_TOKENS for t in vocab_tokens])

        if stars is not None:
            tokens.append(END_TOKEN)
            tokens.append(STAR_TOKENS[stars - 1])

        # pad
        length = len(tokens)
        tokens.extend([EMPTY_TOKEN] * (self.context_size - length))

        return tokens, length

    def decode_context(self, tokens: list[int]) -> str:
        tokens = [x - SPECIAL_TOKENS for x in tokens if x >= 0]
        return self.vocab.DecodeIds(tokens)

    def decode_token(self, token: int) -> str:
        token -= SPECIAL_TOKENS

        piece: str = self.vocab.IdToPiece(token)
        text = self.vocab.DecodeIds(token)
        if piece.startswith("▁"):
            text = " " + text

        return text
