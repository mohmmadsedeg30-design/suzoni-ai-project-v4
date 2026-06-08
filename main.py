#!/usr/bin/env python3
import sys
import argparse

def setup():
    from config import SuzoniConfig
    from tokenizer import SuzoniTokenizer
    from model import SuzoniModel
    c = SuzoniConfig()
    t = SuzoniTokenizer(c.VOCAB_SIZE)
    t._build()
    t.save("tokenizer.json")
    m = SuzoniModel(c)
    m.save("checkpoints/suzoni_initial.pt")
    print("Setup complete!")

def train(dp, r=None):
    from config import SuzoniConfig
    from tokenizer import SuzoniTokenizer
    from model import SuzoniModel
    from trainer import SuzoniTrainer
    c = SuzoniConfig()
    t = SuzoniTokenizer.load("tokenizer.json")
    m = SuzoniModel.load(r, c) if r else SuzoniModel(c)
    tr = SuzoniTrainer(m, t, c, dp)
    tr.train(r)

def integrate(mn, token=None):
    from config import SuzoniConfig
    from tokenizer import SuzoniTokenizer
    from model import SuzoniModel
    from integrator import SuzoniIntegrator
    c = SuzoniConfig()
    m = SuzoniModel.load("checkpoints/suzoni_initial.pt", c)
    t = SuzoniTokenizer.load("tokenizer.json")
    i = SuzoniIntegrator(m, t)
    i.integrate(mn, token)
    i.freeze()
    i.save()

def chat():
    from chat import SuzoniChat
    SuzoniChat().interactive()

def gen(p, m):
    from chat import SuzoniChat
    print(SuzoniChat().chat(p))

def main():
    p = argparse.ArgumentParser(description="Suzoni AI", epilog="Examples:\n  python main.py setup\n  python main.py train --data data.txt\n  python main.py integrate --model gpt2\n  python main.py chat\n  python main.py generate --prompt 'Hi'", formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd")
    s.add_parser("setup")
    t = s.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--resume")
    i = s.add_parser("integrate")
    i.add_argument("--model", required=True)
    i.add_argument("--token")
    s.add_parser("chat")
    g = s.add_parser("generate")
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=100)
    a = p.parse_args()
    if not a.cmd:
        p.print_help()
        sys.exit(1)
    cmds = {
        "setup": setup,
        "train": lambda: train(a.data, a.resume),
        "integrate": lambda: integrate(a.model, a.token),
        "chat": chat,
        "generate": lambda: gen(a.prompt, a.max_tokens)
    }
    cmds[a.cmd]()

if __name__ == "__main__":
    main()
