import torch
from torch import nn
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, src, src_len):
        emb = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden


class DotAttention(nn.Module):
    def forward(self, hidden, encoder_outputs, mask):
        hidden = hidden[-1]
        scores = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attn = DotAttention()
        self.rnn = nn.GRU(emb_dim + enc_hid_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_hid_dim, vocab_size)

    def forward(self, input_tok, hidden, encoder_outputs, mask):
        emb = self.embedding(input_tok)
        context, attn = self.attn(hidden, encoder_outputs, mask)
        rnn_input = torch.cat([emb, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        logits = self.fc_out(output.squeeze(1))
        return logits, hidden, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        return src != self.pad_idx

    @torch.no_grad()
    def translate(self, src, src_len, bos_idx, eos_idx, max_len=50):
        enc_out, hidden = self.encoder(src, src_len)
        mask = self.make_src_mask(src)

        input_tok = torch.tensor([[bos_idx]], device=src.device)
        outputs = []

        for _ in range(max_len):
            logits, hidden, _ = self.decoder(input_tok, hidden, enc_out, mask)
            pred = logits.argmax(1)
            if pred.item() == eos_idx:
                break
            outputs.append(pred.item())
            input_tok = pred.unsqueeze(1)

        return outputs
