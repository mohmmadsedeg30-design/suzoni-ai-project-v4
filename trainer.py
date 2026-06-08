import os
import math
import json
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

class SuzoniTrainer:
    def __init__(self, m, t, c, dp=None):
        self.m = m
        self.t = t
        self.c = c
        self.d = torch.device(c.DEVICE)
        m.to(self.d)

        pd = {pn: p for pn, p in m.named_parameters() if p.requires_grad}
        g = [
            {"params": [p for p in pd.values() if p.dim() >= 2], "weight_decay": 0.1},
            {"params": [p for p in pd.values() if p.dim() < 2], "weight_decay": 0.0}
        ]
        self.o = torch.optim.AdamW(g, lr=c.LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)
        self.s = torch.cuda.amp.GradScaler() if c.DEVICE == "cuda" else None
        self.sc = self._sched()
        self.data = self._load(dp) if dp else None

        self.ld = Path(c.LOG_DIR)
        self.ld.mkdir(exist_ok=True)
        self.log = self.ld / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.tl = []
        self.cd = Path(c.CHECKPOINT_DIR)
        self.cd.mkdir(exist_ok=True)
        self.bv = float("inf")

    def _sched(self):
        def f(s):
            if s < self.c.WARMUP_ITERS:
                return s / self.c.WARMUP_ITERS
            return 0.5 * (1.0 + math.cos(math.pi * (s - self.c.WARMUP_ITERS) / (self.c.MAX_ITERS - self.c.WARMUP_ITERS)))
        return torch.optim.lr_scheduler.LambdaLR(self.o, f)

    def _load(self, p):
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = self.t.encode(text)
        d = torch.tensor(tokens, dtype=torch.long)
        n = int(0.9 * len(d))
        self.train_data = d[:n]
        self.val_data = d[n:]
        return d

    def get_batch(self, s="train"):
        d = self.train_data if s == "train" else self.val_data
        ix = torch.randint(len(d) - self.c.BLOCK_SIZE, (self.c.BATCH_SIZE,))
        return torch.stack([d[i:i + self.c.BLOCK_SIZE] for i in ix]).to(self.d), torch.stack([d[i + 1:i + self.c.BLOCK_SIZE + 1] for i in ix]).to(self.d)

    @torch.no_grad()
    def est(self):
        r = {}
        self.m.eval()
        for s in ["train", "val"]:
            l = torch.zeros(self.c.EVAL_ITERS)
            for k in range(self.c.EVAL_ITERS):
                X, Y = self.get_batch(s)
                if self.s:
                    with torch.cuda.amp.autocast():
                        _, loss = self.m(X, Y)
                else:
                    _, loss = self.m(X, Y)
                l[k] = loss.item()
            r[s] = l.mean().item()
        self.m.train()
        return r

    def train(self, rf=None):
        si = 0
        if rf:
            ck = torch.load(rf)
            self.m.load_state_dict(ck["s"])
            self.o.load_state_dict(ck["o"])
            si = ck.get("step", 0)
        if self.c.COMPILE and hasattr(torch, "compile"):
            self.m = torch.compile(self.m)
        self.m.train()

        for i in tqdm(range(si, self.c.MAX_ITERS), initial=si):
            if i % self.c.EVAL_INTERVAL == 0 and i > 0:
                l = self.est()
                self.tl.append({
                    "i": i,
                    "t": l["train"],
                    "v": l["val"],
                    "lr": self.sc.get_last_lr()[0],
                    "ts": datetime.now().isoformat()
                })
                with open(self.log, "w") as f:
                    json.dump(self.tl, f)
                if l["val"] < self.bv:
                    self.bv = l["val"]
                    self._save("best")
                self._save(i)

            X, Y = self.get_batch("train")
            if self.s:
                with torch.cuda.amp.autocast():
                    _, loss = self.m(X, Y)
                self.s.scale(loss).backward()
                self.s.unscale_(self.o)
                torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0)
                self.s.step(self.o)
                self.s.update()
            else:
                _, loss = self.m(X, Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0)
                self.o.step()

            self.o.zero_grad(set_to_none=True)
            self.sc.step()

        self._save("final")

    def _save(self, step):
        ck = {
            "s": self.m.state_dict(),
            "o": self.o.state_dict(),
            "sc": self.sc.state_dict(),
            "step": step,
            "bv": self.bv,
            "c": {
                "vocab_size": self.c.VOCAB_SIZE,
                "n_embd": self.c.N_EMBD,
                "n_head": self.c.N_HEAD,
                "n_layer": self.c.N_LAYER,
                "block_size": self.c.BLOCK_SIZE
            }
        }
        torch.save(ck, self.cd / f"suzoni_{step}.pt")

    def sample(self, p="الذكاء الاصطناعي", m=100):
        self.m.eval()
        idx = torch.tensor([self.t.encode(p)], dtype=torch.long, device=self.d)
        with torch.no_grad():
            g = self.m.generate(idx, m, 0.8, 40)
            return self.t.decode(g[0].tolist())
