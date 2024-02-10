print("hello world")

import math
import os 
from typing import List, Optional, Tuple, Union
import numpy as np
from transformers import AutoTokenizer
import warnings 

from tinygrad import Tensor, device, nn
from extra.models.bert import BertEmbeddings, BertEncoder


class JinaBertConfig():
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        feed_forward_type="original",
        emb_pooler=None,
        attn_implementation='tiny',
        **kwargs,
    ):
        # super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.feed_forward_type = feed_forward_type
        self.emb_pooler = emb_pooler
        self.attn_implementation = attn_implementation


class JinaBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = Tensor.tanh() 

    def forward(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class BertModel():
    def __init__(self, config, add_pooling_layer=True):
        self.config = config 
        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = JinaBertPooler(config) if add_pooling_layer else None

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        output_value: str = 'sentence_embedding',
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False,
        **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        print("encoding")

        if not self.emb_pooler:
            warnings.warn("No emb_pooler specified, defaulting to mean pooling.")
            self.emb_pooler = 'mean'
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self._name_or_path)
        is_training = self.training
        # self.eval()

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != 'sentence_embedding':
            convert_to_numpy = False
            convert_to_tensor = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_was_string = True

        permutations = np.argsort([-len(i) for i in sentences]) 
        inverse_permutations = np.argsort(permutations)
        sentences = [sentences[idx] for idx in permutations]

        kwargs['padding'] = kwargs.get('padding', True)
        kwargs['max_length'] = kwargs.get('max_length', 8192)
        kwargs['truncation'] = kwargs.get('truncation', True)

        all_embeddings = []

        range_iter = range(0, len(sentences), batch_size)

        for i in range_iter:
            encoded_input = self.tokenizer(
                sentences[i : i + batch_size],
                return_tensors='pt',
                **kwargs
            )
            token_embs = self.forward(**encoded_input)[0]

            token_embs =  token_embs.float()

            assert output_value == 'sentence_embeddings'            

            embeddings = self.mean_pooling(
                token_embs, encoded_input['attention_mask']
            )

            if normalize_embeddings:
                embeddings = Tensor.normalize(embeddings, p=2, dim=1)
            
            if convert_to_numpy:
                raise "convert to numpy not implememted"
                # embeddings =

            all_embeddings = [all_embeddings[idx] for idx in inverse_permutations]
            
            if convert_to_tensor:
                all_embeddings = Tensor.stack(all_embeddings)
            elif convert_to_numpy:
                all_embeddings = np.asanyarray([emb.numpy() for emb in all_embeddings])

            if input_was_string:
                all_embeddings = all_embeddings[0]
            
            # self.train()
            return all_embeddings


    def mean_pooling(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return Tensor.sum(token_embeddings * input_mask_expanded, 1) / Tensor.maximum(input_mask_expanded.sum(1), 1e-9)

    def normalize(self, x: Tensor, p: float = 2, dim: int = 1) -> Tensor:
        return x / Tensor.maximum(x.square().sum(axis=dim).sqrt(), 1e-12)

    def forward(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            encoder_hidden_states: Optional[Tensor] = None,
            encoder_attention_mask: Optional[Tensor] = None,
            past_key_values: Optional[List[Tensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_state: Optional[bool] = None,
            return_dicts: Optional[bool] = None,
    ) -> Tuple[Tensor]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        output_hidden_state = (
            output_hidden_state
            if output_hidden_state is not None
            else self.config.output_hidden_state
        )

        return_dicts = (
            return_dicts
            if return_dicts is not None
            else self.config.return_dicts
        )
        
        # not implementing decoder

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError (
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = Tensor.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )
        
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids: Tensor = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = Tensor.zeros(
                    input_shape, device=device
                )
        
        # TODO might need to implement this
        extended_attention_mask: Tensor = self.get
        encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_state=output_hidden_state,
            return_dicts=return_dicts
        )

        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dicts:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # todo make real
        # return BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        # )