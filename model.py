import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SuzoniAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_head = c.N_HEAD
        self.h = c.N_EMBD // c.N_HEAD
        self.qkv = nn.Linear(c.N_EMBD, 3 * c.N_EMBD, bias=False)
        self.proj = nn.Linear(c.N_EMBD, c.N_EMBD, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(c.N_EMBD, c.N_EMBD // 4),
            nn.GELU(),
            nn.Linear(c.N_EMBD // 4, 1),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(c.DROPOUT)
        self.rdrop = nn.Dropout(c.DROPOUT)
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.h).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.h).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.h).transpose(1, 2)
        g = self.gate(x).unsqueeze(1)
        if self.flash and mask is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            s = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.h))
            m = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            s = s.masked_fill(m == 0, float("-inf"))
            if mask:
                s = s.masked_fill(mask == 0, float("-inf"))
            w = F.softmax(s, dim=-1)
            w = self.drop(w) * g
            out = w @ v
        return self.rdrop(self.proj(out.transpose(1, 2).contiguous().view(B, T, C)))

class SuzoniBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.N_EMBD)
        self.attn = SuzoniAttention(c)
        self.ln2 = nn.LayerNorm(c.N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(c.N_EMBD, 4 * c.N_EMBD),
            nn.GELU(),
            nn.Linear(4 * c.N_EMBD, c.N_EMBD),
            nn.Dropout(c.DROPOUT)
        )
        self.logic = nn.Sequential(
            nn.Linear(c.N_EMBD, c.N_EMBD),
            nn.Tanh()
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.logic(x) * 0.1
        return x + self.mlp(self.ln2(x))

class SuzoniModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.VOCAB_SIZE, c.N_EMBD)
        self.pos = nn.Embedding(c.BLOCK_SIZE, c.N_EMBD)
        self.pers = nn.Embedding(10, c.N_EMBD)
        self.blocks = nn.ModuleList([SuzoniBlock(c) for _ in range(c.N_LAYER)])
        self.ln = nn.LayerNorm(c.N_EMBD)
        self.head = nn.Linear(c.N_EMBD, c.VOCAB_SIZE, bias=False)
        self.drop = nn.Dropout(c.DROPOUT)
        self.apply(self._init)
        p = sum(x.numel() for x in self.parameters())
        print(f"Suzoni: {p / 1e6:.2f}M params | {c.DEVICE}")

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, idx, t=None, pid=0):
        B, T = idx.shape
        x = self.drop(
            self.tok(idx) + 
            self.pos(torch.arange(T, device=idx.device)) + 
            self.pers(torch.tensor([pid], device=idx.device)).unsqueeze(1)
        )
        for b in self.blocks:
            x = b(x)
        x = self.ln(x)
        logits = self.head(x)
        loss = None
        if t is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), t.view(-1), ignore_index=0)
        return logits, loss

    def generate(self, idx, m, t=1.0, k=None, p=None):
        self.eval()
        for _ in range(m):
            x = idx[:, -self.c.BLOCK_SIZE:]
            with torch.no_grad():
                logits, _ = self(x)
                logits = logits[:, -1, :] / t
            if k:
                v, _ = torch.topk(logits, min(k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if p:
                s, _ = torch.sort(logits, descending=True)
                cp = torch.cumsum(F.softmax(s, dim=-1), dim=-1)
                r = cp > p
                r[..., 1:] = r[..., :-1].clone()
                r[..., 0] = 0
                logits[r.scatter(1, _, r)] = float("-inf")
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return idx

    def save(self, path):
        torch.save({
            "s": self.state_dict(),
            "c": {
                "vocab_size": self.c.VOCAB_SIZE,
                "n_embd": self.c.N_EMBD,
                "n_head": self.c.N_HEAD,
                "n_layer": self.c.N_LAYER,
                "block_size": self.c.BLOCK_SIZE,
                "dropout": self.c.DROPOUT
            },
            "v": self.c.VERSION
        }, path)

    @classmethod
    def load(cls, path, c=None):
        ck = torch.load(path, map_location="cpu")
        if c is None:
            from config import SuzoniConfig
            c = SuzoniConfig()
            for k, v in ck["c"].items():
                setattr(c, k.upper(), v)
        m = cls(c)
        m.load_state_dict(ck["s"])
        return m
