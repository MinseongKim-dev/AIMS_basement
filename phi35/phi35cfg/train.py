"""Downstream fine-tuning of Phi-3.5V-Med on one MedVQA train split.

Continues the Med LoRA (r=64): AdamW + cosine, lr 1e-4, batch 64 (micro-batch 1 x 64 accumulation), bf16,
vision encoder frozen, 10 epochs for VQA-RAD / 5 for SLAKE. Checkpoints are chosen on val afterwards
(phi35cfg.evaluate select).

  python -m phi35cfg.train --dataset vqarad --image-root IMAGES --base-dir BASE --output RUN [--attn sdpa]
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from . import model as model_lib
from .data import DATASETS, load_split
from .generation import build_prompt

DEFAULT_EPOCHS = {'vqarad': 10, 'slake': 5}
LORA_RANK = 64


def resolve_epochs(dataset, epochs=None, save_epochs=None):
    """(number of epochs, epochs after which a checkpoint is kept)."""
    epochs = epochs or DEFAULT_EPOCHS[dataset]
    if save_epochs is None:
        keep = set(range(2, epochs + 1, 2)) if epochs >= 10 else set(range(1, epochs + 1))
    else:
        keep = {int(e) for e in save_epochs.split(',')}
    if not keep or not all(1 <= e <= epochs for e in keep):
        raise ValueError(f'--save-epochs must lie in 1..{epochs}')
    return epochs, keep


def keep_checkpoint(epoch, save_epochs, smoke):
    """A --max-steps smoke run keeps its stopping point so that resuming can be checked."""
    return smoke or int(round(epoch)) in save_epochs


def make_collator(processor, image_dir):
    tok = processor.tokenizer

    def collate(rows):
        if len(rows) != 1:
            raise ValueError('Phi-3-V is trained with micro-batch 1')
        r = rows[0]
        with Image.open(Path(image_dir) / r['image']) as im:
            batch = dict(processor(build_prompt(tok, r['question']), [im.convert('RGB')], return_tensors='pt',
                                   qs_highlighted_parts=[]))
        for key in ('attention_mask', 'highlight_attention_mask'):
            batch.pop(key, None)
        answer = tok(f"{r['answer']}<|end|>\n<|endoftext|>", add_special_tokens=False, return_tensors='pt')['input_ids']
        batch['labels'] = torch.cat([torch.full_like(batch['input_ids'], -100), answer], 1)
        batch['input_ids'] = torch.cat([batch['input_ids'], answer], 1)
        return batch

    return collate


def prepare_run(output, config, resume):
    """Record run_config.json and return the checkpoint to resume from (or None)."""
    output = Path(output)
    path = output / 'run_config.json'
    config = json.loads(json.dumps(config, default=str))
    checkpoints = sorted(output.glob('checkpoint-*'), key=lambda d: int(d.name.rsplit('-', 1)[1]))

    def comparable(c):
        rest = {k: v for k, v in c.items() if k not in ('resume', 'max_steps', 'runtime')}
        return {**rest, 'smoke': c.get('max_steps') is not None}

    if resume:
        if not checkpoints or not path.exists():
            raise ValueError(f'--resume: no checkpoint with run_config.json under {output}')
        old, new = comparable(json.loads(path.read_text(encoding='utf-8'))), comparable(config)
        if old != new:
            changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise ValueError(f'--resume with another configuration (differs: {changed})')
        return str(checkpoints[-1])
    if checkpoints:
        raise ValueError(f'{output} already holds checkpoints: pass --resume or use a new --output')
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    return None


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=DATASETS)
    p.add_argument('--image-root', help='VQA-RAD image folder')
    p.add_argument('--data-root', help='SLAKE download folder')
    p.add_argument('--base-dir', required=True)
    p.add_argument('--adapter', default=model_lib.MED_ADAPTER, help='LoRA to continue from')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, help='default: 10 (vqarad), 5 (slake)')
    p.add_argument('--save-epochs', help='e.g. "2,4,6,8,10"')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--warmup-ratio', type=float, default=0.03)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--attn', choices=('eager', 'sdpa'), default='eager', help='sdpa: less memory, faster')
    p.add_argument('--grad-ckpt', action='store_true', help='gradient checkpointing (slower, less memory)')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-steps', type=int, help='smoke run only')
    return p


def main(argv=None):
    parser = build_parser()
    a = parser.parse_args(argv)
    try:
        epochs, save_epochs = resolve_epochs(a.dataset, a.epochs, a.save_epochs)
    except ValueError as e:
        parser.error(str(e))

    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    set_seed(a.seed)
    rows, images = load_split(a.dataset, 'train', a.image_root, a.data_root)
    base = model_lib.prepare_base_dir(a.base_dir)
    if a.attn == 'sdpa':
        model_lib.enable_sdpa(base)                 # before the model class is imported
    model, processor, meta = model_lib.load(base, a.adapter, attn=a.attn, train=True)
    lora = next(iter(model.peft_config.values()))
    if lora.r != LORA_RANK:
        raise ValueError(f'expected LoRA rank {LORA_RANK}, got {lora.r}')
    if a.grad_ckpt:
        model.config.use_cache = False
        model.enable_input_require_grads()
        for name, module in model.named_modules():  # image embeddings are produced here
            if name.endswith('vision_embed_tokens'):
                module.register_forward_hook(lambda m, i, o: o.requires_grad_(True) if o.is_floating_point() else o)
    model.print_trainable_parameters()

    config = {**vars(a), 'epochs': epochs, 'save_epochs': sorted(save_epochs), 'runtime': meta}
    checkpoint = prepare_run(a.output, config, a.resume)
    targs = TrainingArguments(
        output_dir=a.output, num_train_epochs=epochs, max_steps=a.max_steps or -1,
        per_device_train_batch_size=1, gradient_accumulation_steps=a.batch, learning_rate=a.lr,
        lr_scheduler_type='cosine', optim='adamw_torch', weight_decay=a.weight_decay, warmup_ratio=a.warmup_ratio,
        max_grad_norm=a.max_grad_norm, bf16=True, gradient_checkpointing=a.grad_ckpt,
        gradient_checkpointing_kwargs={'use_reentrant': True} if a.grad_ckpt else None,
        save_strategy='epoch', save_total_limit=None, save_only_model=False, remove_unused_columns=False,
        dataloader_num_workers=0, logging_steps=10, report_to='none', seed=a.seed)

    class SaveAtEpochs(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_save = keep_checkpoint(state.epoch, save_epochs, a.max_steps is not None)
            return control

    trainer = Trainer(model=model, args=targs, train_dataset=rows, data_collator=make_collator(processor, images),
                      callbacks=[SaveAtEpochs()])
    if checkpoint:
        # torch >= 2.6 defaults torch.load(weights_only=True), which rejects the numpy RNG state that
        # transformers 4.44 saves in rng_state.pth. These checkpoints are our own.
        load = torch.load
        torch.load = lambda *args, **kw: load(*args, **{**kw, 'weights_only': False})
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(a.output)
    print('Downstream LoRA saved to', a.output)


if __name__ == '__main__':
    main()
