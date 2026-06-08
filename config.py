import torch
from pathlib import Path

class SuzoniConfig:
    NAME = "Suzoni"
    VERSION = "2.0.0"
    PERSONALITY = {
        "name": "Suzoni",
        "traits": ["analytical", "logical", "precise", "witty", "helpful"],
        "speaking_style": "direct and insightful",
        "catchphrase": "Logic is my language.",
        "languages": ["arabic", "english"],
        "knowledge_domains": ["technology", "security", "mathematics", "philosophy"]
    }
    VOCAB_SIZE = 50000
    N_EMBD = 512
    N_HEAD = 8
    N_LAYER = 6
    BLOCK_SIZE = 256
    DROPOUT = 0.1
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    MAX_ITERS = 50000
    EVAL_INTERVAL = 1000
    WARMUP_ITERS = 1000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPILE = torch.cuda.is_available()
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    LOG_DIR = BASE_DIR / "logs"
    for d in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR]:
        d.mkdir(exist_ok=True)
