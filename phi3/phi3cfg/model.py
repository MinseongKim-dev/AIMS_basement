"""Phi-3-Vision (4.2B) + Phi-3V-Med LoRA loader. Phi-3.5 is intentionally not supported."""
import importlib.metadata

import torch

BASE_MODEL = 'microsoft/Phi-3-vision-128k-instruct'
MED_ADAPTER = 'ecoxial2007/Phi-3V-Med'


def patch_clip_for_lora(model):
    """Author's patch (CheX-Phi35V): drop unused CLIP layers. Output is unchanged."""
    embedder = next(m for n, m in model.named_modules() if n.endswith('vision_embed_tokens'))

    def get_img_features(self, img_embeds):
        vm = self.img_processor.vision_model
        h = vm.pre_layrnorm(vm.embeddings(img_embeds))
        return vm.encoder(inputs_embeds=h, output_hidden_states=True).hidden_states[-1][:, 1:]

    layers = embedder.img_processor.vision_model.encoder.layers
    idx = embedder.layer_idx if embedder.layer_idx >= 0 else len(layers) + embedder.layer_idx
    del layers[idx + 1:]
    del embedder.img_processor.vision_model.post_layernorm
    embedder.get_img_features = get_img_features.__get__(embedder)


def load(adapter=MED_ADAPTER, precision='fp16', crops=16, train=False, revision=None):
    """precision: fp16 (T4), bf16 (A100/L4), 4bit (NF4, low memory).

    Eager attention is required: Expert-CFG adds an additive bias to the 4D attention
    mask (Eq. 9), which flash-attention would silently ignore.
    """
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required (Colab: Runtime > Change runtime type > GPU).')
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training

    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    quant = None
    if precision == '4bit':
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                   bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=revision, trust_remote_code=True, torch_dtype=dtype,
        quantization_config=quant, device_map={'': 'cuda:0'},
        _attn_implementation='eager')
    processor = AutoProcessor.from_pretrained(BASE_MODEL, revision=revision,
                                              trust_remote_code=True, num_crops=crops)
    patch_clip_for_lora(model)
    if train:
        if quant:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        else:
            model.requires_grad_(False)
            model.enable_input_require_grads()
        model.config.use_cache = False
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    if train:
        for name, p in model.named_parameters():
            if 'vision_embed_tokens' in name:  # paper: vision encoder frozen
                p.requires_grad_(False)
    else:
        model.eval()
    meta = {'base_model': BASE_MODEL, 'adapter': adapter, 'precision': precision,
            'num_crops': crops, 'gpu': torch.cuda.get_device_name(),
            'versions': {p: importlib.metadata.version(p) for p in
                         ('torch', 'transformers', 'peft', 'accelerate')}}
    return model, processor, meta


def model_dtype(model):
    for p in model.parameters():
        if p.dtype in (torch.float16, torch.bfloat16, torch.float32):
            return p.dtype
    return torch.float16
