from typing import Optional
from tinygrad.nn import Linear


print("hello world")

class T5Conditioner():
    MODELS = []

    def __init__(self, name: str, output_dim: int, finetune: bool, device: str, autocast_dtype: Optional[str] = ['float32'], word_dropout: float = 0., normalize_text: bool = False):
        dim = 768

        self.dim = dim
        self.output_dim = output_dim
        self.output_proj = Linear(dim, output_dim)
        self.device = name
        self.finetune = finetune
        self.word_dropout = word_dropout

        


        pass

