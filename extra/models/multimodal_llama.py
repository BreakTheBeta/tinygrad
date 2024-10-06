import math
import collections
import collections.abc
from typing import Tuple, Union, Optional, Dict, Any
from tinygrad import Tensor, Variable, TinyJit, dtypes, nn, Device
from tinygrad.helpers import getenv
import numpy as np

def complex_mult(A, c, d):
  a,b = A[..., 0:1], A[..., 1:2]
  ro = a*c - b*d
  co = a*d + b*c
  return ro.cat(co, dim=-1)

def apply_rotary_emb(xq:Tensor, xk:Tensor, freqs_cis:Tensor) -> Tuple[Tensor, Tensor]:
  assert freqs_cis.shape[1] == xq.shape[1] == xk.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
  xq = xq.reshape(*xq.shape[0:-1], -1, 2)
  xk = xk.reshape(*xk.shape[0:-1], -1, 2)
  assert len(xq.shape) == len(xk.shape) == len(freqs_cis.shape) == 5
  c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
  xq_out = complex_mult(xq, c, d)
  xk_out = complex_mult(xk, c, d)
  return xq_out.flatten(3), xk_out.flatten(3)

def repeat_kv(x:Tensor, n_rep:int) -> Tensor:
  bs, seqlen, n_kv_heads, head_dim = x.shape
  if n_rep == 1: return x
  # NOTE: this is different from x.repeat((1, 1, n_rep, 1))
  return x.repeat((1, 1, 1, n_rep)).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)


# =================================
# New image stuff for visual llama
# =================================

def build_encoder_attention_mask(
    x: Tensor,
    ar: Tensor,
    ntok: int,
    num_chunks: int,
    n_heads: int,
):
    """
    Build vision encoder attention mask that omits padding tokens.
    """
    masks = []
    for arx in ar:
        mask_i = Tensor.ones((num_chunks, x.shape[2], 1), dtype=x.dtype)
        mask_i[: arx[0] * arx[1], :ntok] = 0
        mask_i = mask_i.view(num_chunks * x.shape[2], -1)
        mask_i = mask_i @ mask_i.T * dtypes.min(x.dtype)
        mask_i = mask_i.unsqueeze(0)
        masks.append(mask_i)
    masks = Tensor.stack(masks).to(x.device).expand(-1, n_heads, -1, -1)
    return masks


def expand_num_tokens_to_mult8(x):
    num_pad_tokens = 8 - (x.shape[-2] % 8)
    if num_pad_tokens == 0:
        return x, 0
    else:
        return (
            Tensor.cat([x, Tensor.zeros( (x.shape[0], x.shape[1], num_pad_tokens, x.shape[-1]), dtype=x.dtype, device=x.device,), ], dim=-2,),
            num_pad_tokens,
        )


def contract_num_tokens_from_mult8(x, num_pad_tokens):
    if num_pad_tokens == 0:
        return x
    return x[:, :, :-num_pad_tokens]


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

def get_strides(shape):
  prod = [1]
  for idx in range(len(shape)-1, -1, -1): prod.append(prod[-1] * shape[idx])
  # something about ints is broken with gpu, cuda
  return Tensor(prod[::-1][1:], dtype=dtypes.int32).unsqueeze(0)

# This is very slow for large arrays, or indices
def _gather(array, indices):
  indices = indices.float().to(array.device)
  reshape_arg = [1]*array.ndim + [array.shape[-1]]
  return Tensor.where(
    indices.unsqueeze(indices.ndim).expand(*indices.shape, array.shape[-1]) == Tensor.arange(array.shape[-1]).reshape(*reshape_arg).expand(*indices.shape, array.shape[-1]),
    array, 0,
  ).sum(indices.ndim)


# TODO, like the other TODO make this work faster with just tinygrad
def npgather(array,indices):
  if isinstance(array, Tensor): array = array.numpy()
  if isinstance(indices, Tensor): indices = indices.numpy()
  if isinstance(indices, list): indices = np.asarray(indices)


def tensor_getitem(tensor, *keys):
  # something about ints is broken with gpu, cuda
  flat_keys = Tensor.stack(*[key.expand((sum(keys)).shape).reshape(-1) for key in keys], dim=1).cast(dtypes.int32)
  strides = get_strides(tensor.shape)
  idxs = (flat_keys * strides).sum(1)
  # TODO add back the flag
  gatherer = npgather #if USE_NP_GATHER else _gather
  return gatherer(tensor.reshape(-1), idxs).reshape(sum(keys).shape)



def _bilinear_interpolate(
  input,  # [N, C, H, W]
  roi_batch_ind,  # [K]
  y,  # [K, PH, IY]
  x,  # [K, PW, IX]
  ymask,  # [K, IY]
  xmask,  # [K, IX]
):
  _, channels, height, width = input.shape
  y = y.clip(min_=0.0, max_=float(height-1))
  x = x.clip(min_=0.0, max_=float(width-1))

  # Tensor.where doesnt work well with int32 data so cast to float32
  y_low = y.cast(dtypes.int32).contiguous().float().contiguous()
  x_low = x.cast(dtypes.int32).contiguous().float().contiguous()

  y_high = Tensor.where(y_low >= height - 1, float(height - 1), y_low + 1)
  y_low = Tensor.where(y_low >= height - 1, float(height - 1), y_low)

  x_high = Tensor.where(x_low >= width - 1, float(width - 1), x_low + 1)
  x_low = Tensor.where(x_low >= width - 1, float(width - 1), x_low)

  ly = y - y_low
  lx = x - x_low
  hy = 1.0 - ly
  hx = 1.0 - lx

  def masked_index(
    y,  # [K, PH, IY]
    x,  # [K, PW, IX]
  ):
    if ymask is not None:
      assert xmask is not None
      y = Tensor.where(ymask[:, None, :], y, 0)
      x = Tensor.where(xmask[:, None, :], x, 0)
    key1 = roi_batch_ind[:, None, None, None, None, None]
    key2 = Tensor.arange(channels, device=input.device)[None, :, None, None, None, None]
    key3 = y[:, None, :, None, :, None]
    key4 = x[:, None, None, :, None, :]
    return tensor_getitem(input,key1,key2,key3,key4)  # [K, C, PH, PW, IY, IX]

  v1 = masked_index(y_low, x_low)
  v2 = masked_index(y_low, x_high)
  v3 = masked_index(y_high, x_low)
  v4 = masked_index(y_high, x_high)

  # all ws preemptively [K, C, PH, PW, IY, IX]
  def outer_prod(y, x):
    return y[:, None, :, None, :, None] * x[:, None, None, :, None, :]

  w1 = outer_prod(hy, hx)
  w2 = outer_prod(hy, lx)
  w3 = outer_prod(ly, hx)
  w4 = outer_prod(ly, lx)

  val = w1*v1 + w2*v2 + w3*v3 + w4*v4
  return val



class ImageFeedForward:
    def __init__(self, dim: int, hidden_dim: int, dropout: float, act_layer = lambda x : x.gelu(),):
        self.c_fc = nn.Linear(dim, hidden_dim, bias=True)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=True)
        self.non_linearity = act_layer
        self.dropout = dropout

    def forward(self, x):
        hidden = nn.linear(x, self.c_fc.weight, self.c_fc.bias)
        hidden = self.non_linearity(hidden)
        hidden = nn.linear(hidden, self.c_proj.weight)
        hidden += self.c_proj.bias
        return hidden


class ImageAttention:
    def __init__(self, dim, head_dim, n_heads,):
        # model_parallel_size = fs_init.get_model_parallel_world_size()

        qkvo_replication = 1
        self.n_kv_heads = n_heads
        self.n_local_heads = n_heads * qkvo_replication
        self.n_local_kv_heads = (self.n_kv_heads * qkvo_replication)
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, qkvo_replication * n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, qkvo_replication * self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, qkvo_replication * self.n_kv_heads * self.head_dim, bias=False,)
        self.wo = nn.Linear(qkvo_replication * n_heads * self.head_dim, dim, bias=False,)
        self.qkvo_replication = qkvo_replication

    def forward(self, x: Tensor, mask: Tensor = None):

        xq, xk, xv = [nn.Linear(x, w) for w in [self.wq.weight, self.wk.weight, self.wv.weight]]

        bs, slen, _ = xq.shape
        
        xq = xq.view(bs, slen, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, xk.shape[1], self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, xv.shape[1], self.n_local_kv_heads, self.head_dim)

        xq, xk, xv = [tensor.transpose(1, 2) for tensor in (xq, xk, xv)]

        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        attn_output = xq.scaled_dot_product_attention(xk, xv, attn_mask=mask, dropout_p=0.0)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bs, slen, -1)

        out = nn.Linear(attn_output, self.wo.weight)
        out = out / self.qkvo_replication
        return out


class ImageTransformerBlock:
    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0, act_layer = lambda x: x.gelu(), gated: bool = False):
        assert d_model % n_head == 0
        self.n_heads = n_head
        self.head_dim = d_model // self.n_heads
        self.attn = ImageAttention(dim=d_model, head_dim=self.head_dim, n_heads=self.n_heads,)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = ImageFeedForward(dim=d_model, hidden_dim=int(mlp_ratio * d_model), dropout=0.0, act_layer=act_layer,)
        self.ln_2 = nn.LayerNorm(d_model)
        self.gated = gated
        if gated:
            self.gate_attn = Tensor.zeros(1)
            self.gate_ffn = Tensor.zeros(1)

    def forward(self, x: Tensor, mask: Tensor = None,):
        _gate_attn = 1 if not self.gated else self.gate_attn.tanh()
        _gate_ffn = 1 if not self.gated else self.gate_ffn.tanh()
        x = x + _gate_attn * self.attn(self.ln_1(x), mask=mask)
        x = x + _gate_ffn * self.mlp(self.ln_2(x))
        return x

class ImageTransformer:
    def __init__(self, width: int, layers: int, heads: int, mlp_ratio: float = 4.0, act_layer = lambda x: x.gelu(), gated: bool = False):
        self.width = width
        self.layers = layers
        # Check for line length later
        self.resblocks = [ImageTransformerBlock(width, heads, mlp_ratio=mlp_ratio, act_layer=act_layer, gated=gated,) for _ in range(self.layers)]

    def forward(self, x: Tensor, return_intermediate=None, mask=None):
        out = []
        for idx, r in enumerate(self.resblocks):
            if return_intermediate is not None and idx in return_intermediate:
                out.append(x)
            x = r(x, mask=mask)
        if return_intermediate is not None:
            return x, Tensor.stack(out, dim=-1)
        return x
        
        
class TilePositionEmbedding:
    def __init__(self, num_tiles: int, width: int, gated: bool = False):
        self.num_tiles = num_tiles
        self.width = width
        self.embedding = Tensor(Tensor.randn(num_tiles, num_tiles, 1, width) / math.sqrt(width))
        self.gated = gated
        if gated:
            self.gate = nn.Parameter(Tensor.zeros(1))

        # TODO LOAD WEIGHTS
        # self._register_load_state_dict_pre_hook(self.load_hook)

    # def load_hook(
    #     self,
    #     state_dict,
    #     prefix,
    #     local_metadata,
    #     strict,
    #     missing_keys,
    #     unexpected_keys,
    #     error_msgs,
    # ):
    #     # load the weights from the checkpoint
    #     embed = state_dict.get(prefix + "embedding")
    #     if embed is not None:
    #         # reshape the weights to the correct shape
    #         nt_old, nt_old, _, w = embed.shape
    #         logging.info(
    #             f"Resizing tile embedding from {nt_old}x{nt_old} to {self.num_tiles}x{self.num_tiles}"
    #         )
    #         embed_new = TilePositionEmbedding._dynamic_resize(embed, self.num_tiles)
    #         # assign the weights to the module
    #         state_dict[prefix + "embedding"] = embed_new

    @staticmethod
    def _dynamic_resize(embed: Tensor, num_tiles: int):
        nt_old, nt_old, _, w = embed.shape
        embed = embed.permute(2, 3, 0, 1)

        embed_new = nn.interpolate(
            embed,
            size=(num_tiles, num_tiles),
            mode="bilinear",
            align_corners=True,
        )
        # reshape the weights to the correct shape
        embed_new = embed_new.permute(2, 3, 0, 1)
        return embed_new

    def forward(self, x: Tensor, ar: Tensor, num_tiles: int = None):
        embed = self.embedding
        if num_tiles is None:
            num_tiles = self.num_tiles
        elif num_tiles > self.num_tiles:
            embed = TilePositionEmbedding._dynamic_resize(self.embedding, num_tiles)
        out_pos_embed = Tensor.zeros(x.shape[0], num_tiles, 1, self.width, device=x.device, dtype=x.dtype)
        for idx, arx in enumerate(ar):
            h, w = arx
            out_pos_embed[idx, : w * h] = embed[:h, :w].reshape(w * h, 1, self.width)
        if self.gated:
            out_pos_embed = out_pos_embed * self.gate.tanh()
        x = x + out_pos_embed
        return x
        

class VisionEncoder:
    def __init__(
        self,
        max_num_tiles: int,
        ckpt_path: str = None,
        image_size: int = 224,
        patch_size: int = 14,
        width: int = 1280,
        layers: int = 32,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        act_layer = lambda x : x.gelu(),
        in_channels: int = 3,
        load_ckpt: bool = False,
        n_global_layers: int = 2,
        global_model: bool = False,
        return_intermediate=None,
    ):
        super().__init__()
        self.global_model = global_model
        self.return_intermediate = return_intermediate
        self.max_num_tiles = max_num_tiles
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False,)
        scale = width**-0.5
        self.class_embedding = Tensor(scale * Tensor.randn(width))
        self.positional_embedding = Tensor(scale * Tensor.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.ln_post = nn.LayerNorm(width)
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = ImageTransformer(width, layers, heads, mlp_ratio, act_layer=act_layer)
        # pre and post tile position embedding
        self.global_transformer = ImageTransformer(width, n_global_layers, heads, mlp_ratio, act_layer=act_layer, gated=True)
        # pre and post tile position embedding

        self.pre_tile_pos_embed = TilePositionEmbedding(num_tiles=max_num_tiles, width=width, gated=True)
        self.post_tile_pos_embed = TilePositionEmbedding(num_tiles=max_num_tiles, width=width, gated=True)

        self.gated_positional_embedding = scale * Tensor.randn(max_num_tiles, max_num_tiles, self.grid_size[0] * self.grid_size[1] + 1, width)
        self.gated_positional_embedding_gate = Tensor.zeros(1)

        # More model loading 
        # TODO 
    #     self._register_load_state_dict_pre_hook(self.load_hook)

    # def load_hook(
    #     self,
    #     state_dict: Dict[str, Any],
    #     prefix: str,
    #     local_metadata: Dict[str, Any],
    #     strict: bool = True,
    #     missing_keys: List[str] = None,
    #     unexpected_keys: List[str] = None,
    #     error_msgs: List[str] = None,
    #     return_state_dict: bool = False,
    # ) -> None:
    #     orig_pos_embed = state_dict.get(prefix + "positional_embedding")
    #     if orig_pos_embed is not None:
    #         new_pos_embed = resize_local_position_embedding(
    #             orig_pos_embed, self.grid_size
    #         )
    #         state_dict[prefix + "positional_embedding"] = new_pos_embed
    #     if hasattr(self, "gated_positional_embedding"):
    #         if prefix + "gated_positional_embedding" not in state_dict:
    #             # resize positional_embedding to fit the new grid size
    #             global_pos_embed = initialize_global_position_embedding_from_local(
    #                 new_pos_embed,
    #                 self.grid_size,
    #                 self.max_num_tiles,
    #                 self.max_num_tiles,
    #             )
    #             state_dict[prefix + "gated_positional_embedding"] = global_pos_embed
    #             state_dict[prefix + "gated_positional_embedding_gate"] = torch.zeros(
    #                 1, dtype=global_pos_embed.dtype
    #             )
    #             logger.info(
    #                 f"Initialized global positional embedding with size {global_pos_embed.size()}"
    #             )
    #         else:
    #             global_pos_embed = resize_global_position_embedding(
    #                 state_dict[prefix + "gated_positional_embedding"],
    #                 self.grid_size,
    #                 self.max_num_tiles,
    #                 self.max_num_tiles,
    #             )
    #             logger.info(
    #                 f"Resized global positional embedding from {state_dict[prefix + 'gated_positional_embedding'].size()} to {global_pos_embed.size()}"
    #             )
    #             state_dict[prefix + "gated_positional_embedding"] = global_pos_embed
    #     if return_state_dict:
    #         return state_dict

    def apply_positional_embedding(self, x, ar):
        out = []
        # apply regular position embedding
        bsz, num_chunks, num_tokens, dim = x.shape
        x = x.view(bsz * num_chunks, num_tokens, dim)
        x = x + self.positional_embedding * (
            1 - self.gated_positional_embedding_gate.tanh()
        )
        x = x.view(bsz, num_chunks, num_tokens, dim)
        for idx, arx in enumerate(ar):
            _pos_embed = self.gated_positional_embedding[: arx[0], : arx[1]]
            _pos_embed = _pos_embed.reshape(arx[0] * arx[1], *_pos_embed.shape[2:])
            x[idx, : arx[0] * arx[1]] += (
                _pos_embed * self.gated_positional_embedding_gate.tanh()
            )
        return x

    def apply_class_embedding(self, x):
        x = Tensor.cat( [ self.class_embedding.to(x.dtype) + Tensor.zeros( x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x,], dim=1,)  
        # shape = [*, grid ** 2 + 1, width]
        return x

    def forward(self, images: Tensor, ar: Tensor) -> Tensor:
        if images.ndim == 5:
            num_concurrent_media = 1
            bsz, num_chunks, nch, w, h = images.shape
        else:
            bsz, num_concurrent_media, num_chunks, nch, w, h = images.shape

        images = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)
        ar = ar.reshape(bsz * num_concurrent_media, 2)

        # patch embedding
        x = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)
        x = self.conv1(x)  # shape = [*, width, grid ** 2]
        _, ntok, dim = x.shape
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)

        # tile embeddings
        x = self.pre_tile_pos_embed(x, ar)
        x = x.reshape(bsz * num_concurrent_media * num_chunks, ntok, dim)

        # apply cls token
        x = self.apply_class_embedding(x)
        ntok += 1

        # apply position embeddings
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)
        x = self.apply_positional_embedding(x, ar)

        x = self.ln_pre(x)
        npad, attn_mask = 0, None
        x, npad = expand_num_tokens_to_mult8(x)
        attn_mask = build_encoder_attention_mask(x, ar, ntok, num_chunks, 1)
        x = x.view(bsz * num_concurrent_media, -1, dim)
        x, int_x = self.transformer(
            x, return_intermediate=self.return_intermediate, mask=attn_mask
        )

        x = self.ln_post(x)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)
        x = self.post_tile_pos_embed(x, ar)
        x = x.reshape(bsz * num_concurrent_media, num_chunks * (ntok + npad), dim)
        x = self.global_transformer(x, mask=attn_mask)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)
        x = contract_num_tokens_from_mult8(x, npad)

        # adding back intermediate layer outputs
        x = x.reshape(bsz, num_concurrent_media, num_chunks, ntok, dim)
        int_x = int_x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, -1)
        int_x = contract_num_tokens_from_mult8(int_x, npad)
        int_x = int_x.reshape(bsz, num_concurrent_media, num_chunks, ntok, -1)
        x = Tensor.cat([x, int_x], dim=-1)
        return x






