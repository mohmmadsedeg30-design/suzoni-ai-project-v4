import json

class SuzoniTokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.special = {
            "<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3,
            "<SPACE>": 4, "<NEWLINE>": 5, "<TAB>": 6,
            "<AR>": 7, "<EN>": 8, "<CODE>": 9
        }
        self._build()

    def _build(self):
        idx = len(self.special)
        for k, v in self.special.items():
            self.vocab[k] = v
            self.inverse_vocab[v] = k

        # Arabic chars
        for i in range(0x0600, 0x0700):
            if idx < self.vocab_size:
                c = chr(i)
                self.vocab[c] = idx
                self.inverse_vocab[idx] = c
                idx += 1

        # English + numbers + symbols
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:\'\"()-[]{}<>@#$%^&*+=_~`|/\\"
        for c in chars:
            if idx < self.vocab_size:
                self.vocab[c] = idx
                self.inverse_vocab[idx] = c
                idx += 1

        # Common words
        words = [
            "ال", "في", "من", "إلى", "على", "هذا", "التي", "الذي", "أن", "لم", "قد", "كان", "كل",
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they",
            "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there",
            "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people",
            "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then",
            "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use",
            "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because",
            "any", "these", "give", "day", "most", "us", "is", "are", "was", "were", "been", "has",
            "had", "did", "does", "doing", "done", "being", "am", "ai", "code", "data", "model",
            "train", "learn", "network", "neural", "deep", "machine", "python", "function", "class",
            "def", "return", "import", "from", "as"
        ]
        for w in words:
            if idx < self.vocab_size and w not in self.vocab:
                self.vocab[w] = idx
                self.inverse_vocab[idx] = w
                idx += 1

    def encode(self, text):
        if not text:
            return []
        tokens = []
        i = 0
        while i < len(text):
            found = False
            for l in [min(4, len(text) - i), 3, 2, 1]:
                s = text[i:i + l]
                if s in self.vocab:
                    tokens.append(self.vocab[s])
                    i += l
                    found = True
                    break
            if not found:
                c = text[i]
                if c == " ":
                    tokens.append(self.special["<SPACE>"])
                elif c in self.vocab:
                    tokens.append(self.vocab[c])
                else:
                    tokens.append(self.special["<UNK>"])
                i += 1
        return tokens

    def decode(self, ids):
        result = []
        for i in ids:
            if i in self.inverse_vocab:
                c = self.inverse_vocab[i]
                if c == "<SPACE>":
                    result.append(" ")
                elif c not in self.special:
                    result.append(c)
        return "".join(result)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab": self.vocab,
                "inverse": {str(k): v for k, v in self.inverse_vocab.items()},
                "special": self.special
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        t = cls(len(d["vocab"]))
        t.vocab = d["vocab"]
        t.inverse_vocab = {int(k): v for k, v in d["inverse"].items()}
        t.special = d["special"]
        return t
