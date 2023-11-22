from typing import Optional, List, Dict
from tinygrad.nn import Linear
from tinygrad.tensor import Tensor
import torch
from transformers import T5EncoderModel, T5Tokenizer 
import random



# Generate audio
# Make a dummy codeblock first 
#
# ['description': (T5)] 
# Output dims = 1536 

# get lm:
# kwargs:
# dim: 1536
# num_heads: 24
# num_layers: 48
# hidden_scale: 4  
# n_q: 8 
# cord: 2048
# dropout: 0 
# 
#   
#   
#   
#   
#   
#   
#   
#   
#   

class T5():
    def __init__(self):
        finetune = False
        self.normalize_text = False
        self.word_dropout = 0.3
        self.training = False
        self.device = "cpu"
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def tokenize(self, x: List[Optional[str]]) -> Dict[str, Tensor]:
        entries: List[str] = [xi if xi is not None else "" for xi in x]
        if self.normalize_text:
            _, _, entries = self.text_normalizer(entries, return_text=True)
        if self.word_dropout > 0. and self.training:
            new_entries = []
            for entry in entries:
                words = [word for word in entry.split(" ") if random.random() >= self.word_dropout]
                new_entries.append(" ".join(words))
            entries = new_entries

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.t5_tokenizer(entries, return_tensors='pt', padding=True).to(self.device)
        mask = inputs['attention_mask']
        mask[empty_idx, :] = 0  # zero-out index where the input is non-existant

        # convert to tiny tensor
        print(inputs)

        newInput = {}
        newInput["input_ids"] = Tensor(inputs['input_ids'].cpu().detach().numpy())
        newInput["attention_mask"] = Tensor(inputs['attention_mask'].cpu().detach().numpy())

        return newInput





class CompressionModel:
    pass








class MusicGen:

    def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel, max_duration: tp.Optional[float] = None):
        self.name = name
        self.compression_model = compression_model
        self.lm = lm
        self.cfg: tp.Optional[omegaconf.DictConfig] = None
        # Just to be safe, let's put everything in eval mode.
        self.compression_model.eval()
        self.lm.eval()

        if hasattr(lm, 'cfg'):
            cfg = lm.cfg
            assert isinstance(cfg, omegaconf.DictConfig)
            self.cfg = cfg

        if self.cfg is not None:
            self.compression_model = get_wrapped_compression_model(self.compression_model, self.cfg)

        if max_duration is None:
            if self.cfg is not None:
                max_duration = lm.cfg.dataset.segment_duration  # type: ignore
            else:
                raise ValueError("You must provide max_duration when building directly MusicGen")
        assert max_duration is not None
        self.max_duration: float = max_duration
        self.device = next(iter(lm.parameters())).device

        self.generation_params: dict = {}
        self.set_generation_params(duration=15)  # 15 seconds by default
        self._progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None
        if self.device.type == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
        else:
            self.autocast = TorchAutocast(
                enabled=True, device_type=self.device.type, dtype=torch.float16)


    # MOOSE 4
    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            descriptions: tp.Sequence[tp.Optional[str]],
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (torch.Tensor, optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        # MOOSE 2
        attributes = [
            ConditioningAttributes(text={'description': description})
            for description in descriptions]

        if melody_wavs is None:
            for attr in attributes:
                # MOOSE 3
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    sample_rate=[self.sample_rate],
                    path=[None])
        else:
            if 'self_wav' not in self.lm.condition_provider.conditioners:
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None])
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody[None].to(device=self.device),
                        torch.tensor([melody.shape[-1]], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None],
                    )

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens




































t = T5()
a = t.tokenize(["Your mum is fat, I farted in your dad's face and he didn't like it"])






print(a)