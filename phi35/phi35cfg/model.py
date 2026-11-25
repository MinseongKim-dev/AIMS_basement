"""Phi-3.5-vision-instruct + LoRA adapter + the authors' processor (which builds the highlight mask)."""
import importlib.metadata
import shutil
import sys
from pathlib import Path

from . import REPO_ROOT

BASE_MODEL = 'microsoft/Phi-3.5-vision-instruct'
MED_ADAPTER = 'ecoxial2007/Phi-3.5V-Med'
ASSETS = REPO_ROOT / 'phi35' / 'assets'            # filled by tools/fetch_author_assets.py
PROCESSOR_FILES = ('processing_phi3_v_cfg.py', 'processor_config.json')


def prepare_base_dir(base_dir):
    """Download the base model if needed and put the authors' processor files over the hub ones."""
    base = Path(base_dir)
    highlight_module = ASSETS / 'author_src' / 'src' / 'highlighter_modules' / 'utils.py'
    missing = [p for p in [ASSETS / f for f in PROCESSOR_FILES] + [highlight_module] if not p.exists()]
    if missing:
        raise FileNotFoundError(f'author files missing ({missing[0].name}): run python tools/fetch_author_assets.py')
    if not (base / 'config.json').exists():
        from huggingface_hub import snapshot_download
        snapshot_download(BASE_MODEL, local_dir=str(base))
    for name in PROCESSOR_FILES:                    # always: a download restores the hub processor
        shutil.copyfile(ASSETS / name, base / name)
    source = str(ASSETS / 'author_src')             # the processor imports src.highlighter_modules
    if source not in sys.path:
        sys.path.insert(0, source)
    return base


def enable_sdpa(base_dir):
    """The remote code ships Phi3SdpaAttention but sets `_supports_sdpa = False`; flip it (training only).
    Must run before the model class is first imported."""
    path = Path(base_dir) / 'modeling_phi3_v.py'
    source = path.read_text(encoding='utf-8')
    if '_supports_sdpa = False' in source:
        path.write_text(source.replace('_supports_sdpa = False', '_supports_sdpa = True'), encoding='utf-8',
                        newline='\n')
    elif '_supports_sdpa = True' not in source:
        raise RuntimeError(f'{path}: no _supports_sdpa flag found')


def resolve_adapter(name_or_path):
    """Folder holding adapter_config.json, from a local directory or a Hugging Face repository."""
    root = Path(name_or_path)
    if not root.is_dir():
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(str(name_or_path)))
    if (root / 'adapter_config.json').exists():
        return str(root)
    found = sorted(root.rglob('adapter_config.json'))
    if len(found) != 1:
        raise FileNotFoundError(f'{name_or_path}: expected one adapter_config.json, found {len(found)}')
    return str(found[0].parent)


def load(base_dir, adapter, num_crops=16, attn='eager', train=False):
    """Return (model, processor, meta). bf16. Inference is always eager: the CFG hooks add a bias to the
    4D attention mask. attn='sdpa' is an opt-in for training (call enable_sdpa first)."""
    if attn not in ('eager', 'sdpa'):
        raise ValueError(f'unknown attention implementation: {attn}')
    if attn == 'sdpa' and not train:
        raise ValueError('sdpa is for training only; inference must stay eager')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('a CUDA GPU is required')
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

    base_path = prepare_base_dir(base_dir)
    adapter_path = resolve_adapter(adapter)
    processor = AutoProcessor.from_pretrained(str(base_path), trust_remote_code=True, num_crops=num_crops)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path), torch_dtype=torch.bfloat16, trust_remote_code=True,
        _attn_implementation=attn, device_map={'': 'cuda:0'})
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=train)
    if train:
        for name, p in model.named_parameters():
            if 'vision_embed_tokens' in name:       # paper: vision encoder frozen
                p.requires_grad_(False)
    else:
        model.eval()
    model.generation_config = GenerationConfig.from_pretrained(str(base_path))
    meta = {'base_model': BASE_MODEL, 'adapter': str(adapter), 'precision': 'bf16', 'attn': attn,
            'num_crops': num_crops, 'gpu': torch.cuda.get_device_name(),
            'python': sys.version.split()[0],
            'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'accelerate')}}
    return model, processor, meta
