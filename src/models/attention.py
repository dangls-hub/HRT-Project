import torch
import torch.nn as nn
from typing import Tuple

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """
        Khởi tạo cơ chế Bahdanau (Additive) Attention.
        - encoder_dim: số kênh đặc trưng của encoder đầu ra (ví dụ: output_h * channels).
        - decoder_dim: kích thước hidden state của decoder.
        - attention_dim: chiều không gian chiếu ẩn dùng để tính toán attention scores.
        """
        super().__init__()
        self.W_enc = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_dec = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.v_att = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        decoder_hidden: (B, decoder_dim) — Hidden state của decoder tại bước hiện tại.
        encoder_outputs: (B, seq_len, encoder_dim) — Chuỗi đầu ra của encoder.
        
        Trả về:
        - context_vector: (B, encoder_dim) — Vector ngữ cảnh kết hợp.
        - attn_weights: (B, seq_len) — Phân phối trọng số attention trên các bước sequence.
        """
        # enc_proj: (B, seq_len, attention_dim)
        enc_proj = self.W_enc(encoder_outputs)
        
        # dec_proj: (B, 1, attention_dim)
        dec_proj = self.W_dec(decoder_hidden).unsqueeze(1)
        
        # energy: (B, seq_len, attention_dim)
        energy = torch.tanh(enc_proj + dec_proj)
        
        # scores: (B, seq_len)
        scores = self.v_att(energy).squeeze(-1)
        
        # attn_weights: (B, seq_len)
        attn_weights = torch.softmax(scores, dim=1)
        
        # context_vector: (B, encoder_dim)
        # Chuyển attn_weights thành (B, 1, seq_len) nhân với (B, seq_len, encoder_dim)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context_vector, attn_weights