import ctypes

import numpy as np
import torch
from llama_cpp import Llama
from llama_cpp import llama_cpp as lc
from transformers import PretrainedConfig, PreTrainedModel

# llama.cpp extended API: output the residual-stream hidden states feeding into
# layer `lid` (lid 0 = embedding output). Not exposed by llama-cpp-python, so we
# call it directly through the loaded CDLL. The symbols live in llama-ext.h which
# lacks extern "C", so we try the plain name first and fall back to the known
# Itanium-C++-mangled names.
_MANGLED_SET = "_Z30llama_set_embeddings_layer_inpP13llama_contextjb"
_MANGLED_GET = "_Z30llama_get_embeddings_layer_inpP13llama_contextj"


class LlamaQwen3TextEncoder(PreTrainedModel):
    """FLUX.2 [klein] Qwen3-4B text encoder served by llama.cpp in native q4.

    The GGUF stays packed in VRAM (~2.5 GB, no bf16 dequantization). Per-token
    hidden states are extracted at the layers FLUX.2 expects (9, 18, 27) and
    stacked into `prompt_embeds` that `Flux2KleinPipeline` consumes via its
    `prompt_embeds=` argument.

    Subclasses `PreTrainedModel` only so that diffusers accepts it as the
    pipeline's `text_encoder` component (it is never actually used as a
    transformers model).
    """

    config_class = PretrainedConfig

    def __init__(self, model_path, tokenizer, layers=(9, 18, 27), n_ctx=512, device="cuda"):
        super().__init__(PretrainedConfig())
        self.tokenizer = tokenizer
        self.layers = tuple(layers)
        self._device = device
        self._dtype = torch.bfloat16

        self.llm = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            verbose=False,
        )
        self.n_embd = self.llm.n_embd()
        self._bind_ext()
        for lid in self.layers:
            self._set_layer_inp_fn(self.llm._ctx.ctx, lid, True)

    def _bind_ext(self):
        lib = lc._lib
        set_fn = getattr(lib, "llama_set_embeddings_layer_inp", None)
        if set_fn is None:
            set_fn = getattr(lib, _MANGLED_SET)
        get_fn = getattr(lib, "llama_get_embeddings_layer_inp", None)
        if get_fn is None:
            get_fn = getattr(lib, _MANGLED_GET)

        set_fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_bool]
        set_fn.restype = None
        get_fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        get_fn.restype = ctypes.POINTER(ctypes.c_float)
        self._set_layer_inp_fn = set_fn
        self._get_layer_inp_fn = get_fn

    @property
    def device(self):
        return torch.device(self._device)

    @property
    def dtype(self):
        return self._dtype

    def to(self, *args, **kwargs):
        return self

    def encode_prompt(self, prompt, max_length=512):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self.tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
        n = len(ids)

        self.llm._ctx.kv_cache_clear()
        self.llm._batch.reset()
        self.llm._batch.add_sequence(ids, 0, True)
        self.llm._ctx.decode(self.llm._batch)

        ctx = self.llm._ctx.ctx
        tensors = []
        for lid in self.layers:
            ptr = self._get_layer_inp_fn(ctx, lid)
            arr = np.ctypeslib.as_array(ptr, shape=(n * self.n_embd,)).reshape(n, self.n_embd)
            tensors.append(torch.from_numpy(arr.copy()))

        out = torch.stack(tensors, dim=1)  # [L, 3, D]
        out = out.permute(1, 0, 2).reshape(1, n, 3 * self.n_embd)  # [1, L, 3D]
        return out.to(device=self._device, dtype=self._dtype)

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "LlamaQwen3TextEncoder must be driven via encode_prompt() and "
            "pipe(prompt_embeds=...); the padded-token pipeline path is not supported."
        )
