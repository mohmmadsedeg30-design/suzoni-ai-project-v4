#!/usr/bin/env python3
import sys
import json
import torch
import argparse

class SuzoniChat:
    def __init__(self, mp="checkpoints/suzoni_final.pt", tp="tokenizer.json"):
        from model import SuzoniModel
        from tokenizer import SuzoniTokenizer
        from config import SuzoniConfig

        self.c = SuzoniConfig()
        self.t = SuzoniTokenizer.load(tp)
        self.m = SuzoniModel.load(mp, self.c)
        self.m.eval()
        self.d = torch.device(self.c.DEVICE)
        self.h = []
        self.sp = f"You are {self.c.PERSONALITY['name']}. Style: {self.c.PERSONALITY['speaking_style']}. Traits: {', '.join(self.c.PERSONALITY['traits'])}. Catchphrase: {self.c.PERSONALITY['catchphrase']}. Respond helpfully."

    def gen(self, p, m=200, t=0.8, tp=0.9):
        idx = torch.tensor([self.t.encode(p)], dtype=torch.long, device=self.d)
        with torch.no_grad():
            g = self.m.generate(idx, m, t, 50, tp)
            return self.t.decode(g[0].tolist())

    def chat(self, msg):
        ctx = self.sp + "\n\n" + "".join(f"User: {x['u']}\nSuzoni: {x['s']}\n\n" for x in self.h[-5:]) + f"User: {msg}\nSuzoni:"
        r = self.gen(ctx)
        if "Suzoni:" in r:
            r = r.split("Suzoni:")[-1].strip()
        self.h.append({"u": msg, "s": r})
        return r

    def interactive(self):
        print("\n" + "=" * 60 + "\nSuzoni AI\n" + "=" * 60 + "\nCommands: /save /clear /config /exit\n" + "=" * 60 + "\n")
        while True:
            try:
                i = input("You: ").strip()
                if not i:
                    continue
                if i == "/exit":
                    print("Suzoni: Goodbye!")
                    break
                if i == "/clear":
                    self.h = []
                    print("Cleared")
                    continue
                if i == "/save":
                    self.save()
                    continue
                if i == "/config":
                    print(f"\n{self.c.NAME} v{self.c.VERSION}\nParams: {sum(p.numel() for p in self.m.parameters()) / 1e6:.2f}M\nDevice: {self.c.DEVICE}\nHistory: {len(self.h)}\n")
                    continue
                print(f"Suzoni: {self.chat(i)}\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def save(self, p="chat_history.json"):
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"c": {"n": self.c.NAME, "v": self.c.VERSION}, "h": self.h}, f, ensure_ascii=False)
        print(f"Saved {p}")

def main():
    p = argparse.ArgumentParser(description="Suzoni")
    p.add_argument("--model", default="checkpoints/suzoni_final.pt")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--prompt")
    args = p.parse_args()

    c = SuzoniChat(args.model, args.tokenizer)
    if args.prompt:
        print(c.chat(args.prompt))
    else:
        c.interactive()

if __name__ == "__main__":
    main()
