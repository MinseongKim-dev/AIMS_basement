"""Downstream fine-tuning of Phi3V-Med on one MedVQA train split (Sec. 4.3).

Paper: LoRA rank 64 only, AdamW + cosine, lr 1e-4, batch 64, 10 epochs, vision encoder frozen.
Starts from the released Phi-3V-Med LoRA (ecoxial2007/Phi-3V-Med, r=64, alpha=128).
Phi-3-V supports micro-batch 1, so batch 64 = 1 x 64 gradient accumulation.

  python -m phi3cfg.train --dataset-dir DATA/vqa_rad --output runs/phi3v_vqa_rad
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .data import load_data
from .generation import build_prompt
from .model import MED_ADAPTER, load, model_dtype


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--adapter', default=MED_ADAPTER)
    p.add_argument('--precision', choices=('fp16', 'bf16', '4bit'), default='fp16')
    p.add_argument('--crops', type=int, default=16, help='lower (e.g. 4) if T4 runs out of memory')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    # Not stated in the paper; defaults can be taken from the released adapter's training_args.bin
    p.add_argument('--warmup-ratio', type=float, default=0.03)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--save-epochs', help='e.g. "2,4,6,8,10": checkpoint only at these epochs and keep them all '
                                         '(for validation-based selection). Default: every epoch, keep last 2')
    p.add_argument('--no-grad-ckpt', action='store_true', help='disable gradient checkpointing (faster, more memory)')
    args = p.parse_args()
    save_epochs = {int(e) for e in args.save_epochs.split(',')} if args.save_epochs else None
    if save_epochs and not all(1 <= e <= args.epochs for e in save_epochs):
        p.error(f'--save-epochs must lie in 1..{args.epochs}')

    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    set_seed(args.seed)
    ds_dir = Path(args.dataset_dir)
    train = load_data(ds_dir / 'train.jsonl', 'train')
    model, processor, meta = load(args.adapter, args.precision, args.crops, train=True)
    lora = next(iter(model.peft_config.values()))
    assert lora.r == 64, f'expected LoRA rank 64, got {lora.r}'
    for prm in model.parameters():  # AMP needs fp32 master weights for trainable params
        if prm.requires_grad:
            prm.data = prm.data.float()
    model.print_trainable_parameters()
    tok = processor.tokenizer
    dtype = model_dtype(model)

    def collate(rows):
        assert len(rows) == 1, 'Phi-3-V supports micro-batch 1'
        r = rows[0]
        with Image.open(ds_dir / 'images' / r['image']) as im:
            batch = processor(build_prompt(tok, r['question']), [im.convert('RGB')], return_tensors='pt')
        ans = tok(f"{r['answer']}<|end|>\n<|endoftext|>", add_special_tokens=False,
                  return_tensors='pt')['input_ids']
        batch['labels'] = torch.cat([torch.full_like(batch['input_ids'], -100), ans], 1)
        batch['input_ids'] = torch.cat([batch['input_ids'], ans], 1)
        batch['attention_mask'] = torch.ones_like(batch['input_ids'])
        batch['pixel_values'] = batch['pixel_values'].to(dtype)
        return dict(batch)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'run_config.json').write_text(json.dumps({**vars(args), 'runtime': meta}, indent=2))
    targs = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs, per_device_train_batch_size=1,
        gradient_accumulation_steps=args.batch, learning_rate=args.lr, lr_scheduler_type='cosine',
        optim='adamw_torch', weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        fp16=args.precision != 'bf16', bf16=args.precision == 'bf16',
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={'use_reentrant': False} if not args.no_grad_ckpt else None,
        remove_unused_columns=False, dataloader_num_workers=2, logging_steps=5,
        save_strategy='epoch', save_total_limit=None if save_epochs else 2,
        report_to='none', seed=args.seed)

    class SaveAtEpochs(TrainerCallback):
        """Runs after the default flow callback: keep its per-epoch save only at the chosen epochs."""
        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_save = int(round(state.epoch)) in save_epochs
            return control

    trainer = Trainer(model=model, args=targs, train_dataset=train, data_collator=collate,
                      callbacks=[SaveAtEpochs()] if save_epochs else None)
    if args.resume:
        # torch>=2.6 defaults torch.load(weights_only=True), which rejects the numpy RNG state that
        # transformers 4.44 saves in rng_state.pth. These checkpoints are our own, so load them fully.
        _load = torch.load
        torch.load = lambda *a, **k: _load(*a, **{**k, 'weights_only': False})
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(str(out))  # adapter_config.json + adapter_model.safetensors
    print('Downstream LoRA saved to', out, '-> use as --adapter for phi3cfg.infer')


if __name__ == '__main__':
    main()
