import torch

class SuzoniIntegrator:
    def __init__(self, m, t):
        self.m = m
        self.t = t
        self.log = []
        self.d = next(m.parameters()).device

    def integrate(self, name, token=None):
        from transformers import AutoModelForCausalLM
        print(f"Loading {name}...")
        kwargs = {"token": token} if token else {}
        try:
            hf = AutoModelForCausalLM.from_pretrained(name, **kwargs)
            st = hf.state_dict()
        except Exception as e:
            print(f"Error: {e}")
            return None

        if "transformer.wte.weight" in st:
            return self._gpt2(st, name)
        elif "model.embed_tokens.weight" in st:
            return self._llama(st, name)
        elif "transformer.word_embeddings.weight" in st:
            return self._bloom(st, name)
        else:
            return self._auto(st, name)

    def _gpt2(self, st, name):
        m = {
            "transformer.wte.weight": "tok.weight",
            "transformer.wpe.weight": "pos.weight",
            "transformer.ln_f.weight": "ln.weight",
            "transformer.ln_f.bias": "ln.bias"
        }
        for i in range(min(12, self.m.c.N_LAYER)):
            m[f"transformer.h.{i}.ln_1.weight"] = f"blocks.{i}.ln1.weight"
            m[f"transformer.h.{i}.ln_1.bias"] = f"blocks.{i}.ln1.bias"
            m[f"transformer.h.{i}.attn.c_attn.weight"] = f"blocks.{i}.attn.qkv.weight"
            m[f"transformer.h.{i}.attn.c_attn.bias"] = f"blocks.{i}.attn.qkv.bias"
            m[f"transformer.h.{i}.attn.c_proj.weight"] = f"blocks.{i}.attn.proj.weight"
            m[f"transformer.h.{i}.attn.c_proj.bias"] = f"blocks.{i}.attn.proj.bias"
            m[f"transformer.h.{i}.ln_2.weight"] = f"blocks.{i}.ln2.weight"
            m[f"transformer.h.{i}.ln_2.bias"] = f"blocks.{i}.ln2.bias"
            m[f"transformer.h.{i}.mlp.c_fc.weight"] = f"blocks.{i}.mlp.0.weight"
            m[f"transformer.h.{i}.mlp.c_fc.bias"] = f"blocks.{i}.mlp.0.bias"
            m[f"transformer.h.{i}.mlp.c_proj.weight"] = f"blocks.{i}.mlp.2.weight"
            m[f"transformer.h.{i}.mlp.c_proj.bias"] = f"blocks.{i}.mlp.2.bias"
        return self._tx(st, m, name)

    def _llama(self, st, name):
        m = {"model.embed_tokens.weight": "tok.weight", "model.norm.weight": "ln.weight"}
        for i in range(min(32, self.m.c.N_LAYER)):
            m[f"model.layers.{i}.input_layernorm.weight"] = f"blocks.{i}.ln1.weight"
            m[f"model.layers.{i}.post_attention_layernorm.weight"] = f"blocks.{i}.ln2.weight"
            m[f"model.layers.{i}.self_attn.q_proj.weight"] = f"blocks.{i}.attn.qkv.weight"
            m[f"model.layers.{i}.self_attn.k_proj.weight"] = f"blocks.{i}.attn.qkv.weight"
            m[f"model.layers.{i}.self_attn.v_proj.weight"] = f"blocks.{i}.attn.qkv.weight"
            m[f"model.layers.{i}.self_attn.o_proj.weight"] = f"blocks.{i}.attn.proj.weight"
            m[f"model.layers.{i}.mlp.gate_proj.weight"] = f"blocks.{i}.mlp.0.weight"
            m[f"model.layers.{i}.mlp.up_proj.weight"] = f"blocks.{i}.mlp.0.weight"
            m[f"model.layers.{i}.mlp.down_proj.weight"] = f"blocks.{i}.mlp.2.weight"
        return self._tx(st, m, name)

    def _bloom(self, st, name):
        m = {
            "transformer.word_embeddings.weight": "tok.weight",
            "transformer.word_embeddings_layernorm.weight": "ln.weight",
            "transformer.word_embeddings_layernorm.bias": "ln.bias",
            "transformer.ln_f.weight": "ln.weight",
            "transformer.ln_f.bias": "ln.bias"
        }
        for i in range(min(24, self.m.c.N_LAYER)):
            m[f"transformer.h.{i}.input_layernorm.weight"] = f"blocks.{i}.ln1.weight"
            m[f"transformer.h.{i}.input_layernorm.bias"] = f"blocks.{i}.ln1.bias"
            m[f"transformer.h.{i}.self_attention.query_key_value.weight"] = f"blocks.{i}.attn.qkv.weight"
            m[f"transformer.h.{i}.self_attention.query_key_value.bias"] = f"blocks.{i}.attn.qkv.bias"
            m[f"transformer.h.{i}.self_attention.dense.weight"] = f"blocks.{i}.attn.proj.weight"
            m[f"transformer.h.{i}.self_attention.dense.bias"] = f"blocks.{i}.attn.proj.bias"
            m[f"transformer.h.{i}.post_attention_layernorm.weight"] = f"blocks.{i}.ln2.weight"
            m[f"transformer.h.{i}.post_attention_layernorm.bias"] = f"blocks.{i}.ln2.bias"
            m[f"transformer.h.{i}.mlp.dense_h_to_4h.weight"] = f"blocks.{i}.mlp.0.weight"
            m[f"transformer.h.{i}.mlp.dense_h_to_4h.bias"] = f"blocks.{i}.mlp.0.bias"
            m[f"transformer.h.{i}.mlp.dense_4h_to_h.weight"] = f"blocks.{i}.mlp.2.weight"
            m[f"transformer.h.{i}.mlp.dense_4h_to_h.bias"] = f"blocks.{i}.mlp.2.bias"
        return self._tx(st, m, name)

    def _auto(self, st, name):
        ss = self.m.state_dict()
        c = {}
        for ek, et in st.items():
            for sk, st_ in ss.items():
                if et.shape == st_.shape:
                    c[ek] = sk
                    break
        return self._tx(st, c, name)

    def _tx(self, ext, m, src):
        ss = self.m.state_dict()
        t = 0
        for ek, sk in m.items():
            if ek in ext and sk in ss:
                ss[sk] = ext[ek].to(self.d)
                t += 1
        self.m.load_state_dict(ss)
        self.log.append({"src": src, "t": t, "tot": len(ss), "p": t / len(ss) * 100})
        print(f"Transferred {t}/{len(ss)} ({t / len(ss) * 100:.1f}%)")
        return self.m

    def freeze(self, f=True):
        for n, p in self.m.named_parameters():
            p.requires_grad = not f
        self.m.head.weight.requires_grad = True
        self.m.pers.weight.requires_grad = True
        print(f"External layers {'frozen' if f else 'unfrozen'}")

    def blend(self, paths, w=None):
        if w is None:
            w = [1.0 / len(paths)] * len(paths)
        s = [torch.load(p, map_location="cpu")["s"] if "s" in torch.load(p, map_location="cpu") else torch.load(p, map_location="cpu") for p in paths]
        b = {k: sum(st[k] * wt for st, wt in zip(s, w)) for k in s[0].keys()}
        self.m.load_state_dict(b)
        print("Blended")
        return self.m

    def save(self, p=None):
        if p is None:
            p = self.m.c.CHECKPOINT_DIR / "suzoni_int.pt"
        self.m.save(p)
        return p
