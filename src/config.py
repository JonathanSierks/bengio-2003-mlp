from dataclasses import dataclass

@dataclass
class Config:
    # Device
    device: str | None = None  # "cuda", "mps", "cpu", None = auto

    # Model hyperparameters
    model: str = "mlp"         # "mlp" oder "baseline"
    emb_dim: int = 30           # Embedding Dimension
    hidden_dim: int = 100       # Hidden Layer Größe
    context_size: int = 5       # n-gram Kontext

    # Training hyperparameters
    epochs: int = 20
    batch_size: int = 32
    epsilon_t: float = 0.1     # Lernrate
    lr_decay: float = 1e-8
    weight_decay: float = 1e-4

    # Data
    shuffle: bool = True