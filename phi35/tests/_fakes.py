"""A tiny stand-in for the Phi-3.5-V model and processor, with the module names the CFG hooks target."""
import torch
from torch import nn

V = 10
END, SPECIAL, EOS = 9, 8, 7                      # <|end|>, another special token, tokenizer eos
WORDS = {'a': 2, 'b': 3, '<s>': SPECIAL}
PROMPT_IDS = [1, -1, 5, 6]                       # -1 is an image placeholder; position 2 is the highlightable token


class Tokenizer:
    eos_token_id = EOS
    all_special_ids = [END, SPECIAL, EOS]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<user>{messages[0]['content']}<end><assistant>"

    def convert_tokens_to_ids(self, token):
        return END

    def __call__(self, text, add_special_tokens=False, return_tensors='pt'):
        return {'input_ids': torch.tensor([[WORDS[w] for w in text.split()]], dtype=torch.long)}


class Processor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.calls = []

    def __call__(self, prompt, images, return_tensors='pt', qs_highlighted_parts=None):
        self.calls.append((prompt, qs_highlighted_parts))
        ids = torch.tensor([PROMPT_IDS], dtype=torch.long)
        mask = torch.zeros_like(ids)
        if 'finding' in (qs_highlighted_parts or []):
            mask[0, 2] = 1
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'highlight_attention_mask': mask,
                'pixel_values': torch.zeros(1, 1, 3, 2, 2), 'image_sizes': torch.tensor([[2, 2]])}

    def batch_decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        names = {v: k for k, v in WORDS.items()}
        return [' ' + ' '.join(names[i] for i in row.tolist() if i not in Tokenizer.all_special_ids) + '. ' for row in ids]


class Attention(nn.Module):
    def forward(self, hidden, attention_mask=None):
        self.seen = attention_mask
        return hidden


class Embed(nn.Module):
    def forward(self, input_ids):
        self.seen = torch.ones(input_ids.shape[0], input_ids.shape[1], 2)
        return self.seen


class Spy(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x):
        self.seen = x
        return x


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()


class Out:
    def __init__(self, logits, past):
        self.logits, self.past_key_values = logits, past


class Model(nn.Module):
    """Next token at sequence length P+k is script[k]. `table` overrides the logits of given positions."""

    def __init__(self, script=(2, 3, END), table=None, positional=False, fail=False):
        super().__init__()
        self.vision_embed_tokens = Embed()
        self.after_embed = Spy()
        self.layers = nn.ModuleList([Layer()])
        self.weight = nn.Embedding(V, 2)
        self.script, self.table, self.positional, self.fail = list(script), table or {}, positional, fail
        self.calls, self.embeds, self.masks = [], [], []

    def get_input_embeddings(self):
        return self.weight

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_sizes=None,
                position_ids=None, past_key_values=None, use_cache=False):
        B, n = input_ids.shape
        past = past_key_values or 0
        total = past + n
        self.calls.append({'B': B, 'n': n, 'past': past, 'position_ids': position_ids,
                           'pixels': pixel_values is not None, 'attention_mask': attention_mask})
        hidden = self.after_embed(self.vision_embed_tokens(input_ids))
        self.embeds.append(hidden.clone())
        mask4d = torch.zeros(B, 1, n, total)
        attn = self.layers[0].self_attn
        if self.positional:
            attn(hidden, mask4d)
        else:
            attn(hidden, attention_mask=mask4d)
        self.masks.append(attn.seen.clone())
        input_ids.clamp_(min=0)                  # like Phi-3-V, which edits image placeholders in place
        if self.fail:
            raise RuntimeError('forward failed')
        logits = torch.zeros(B, n, V)
        P = len(PROMPT_IDS)
        for j in range(n):
            pos = past + j
            if pos in self.table:
                logits[:, j] = self.table[pos]
            elif pos >= P - 1:
                k = min(pos - (P - 1), len(self.script) - 1)
                logits[:, j, self.script[k]] = 10.0
        return Out(logits, total if use_cache else None)
