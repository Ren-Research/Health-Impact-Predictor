import torch
import torch.nn as nn

# Define the Transformer model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads, num_encoder_layers=1, num_decoder_layers=1, dropout=dropout
        )
        self.fc_out = nn.Linear(embed_dim, input_dim)

    def forward(self, src, tgt):
        # Embed the input and target
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)

        # Transpose toclass TimeSeriesTransformer(nn.Module):
        #     def __init__(self, input_dim, embed_dim, num_heads, dropout=0.1):
        #         super(TimeSeriesTransformer, self).__init__()
        #         self.embedding = nn.Linear(input_dim, embed_dim)
        #         self.transformer = nn.Transformer(
        #             d_model=embed_dim, nhead=num_heads, num_encoder_layers=1, num_decoder_layers=1, dropout=dropout
        #         )
        #         self.fc_out = nn.Linear(embed_dim, input_dim)
        #
        #     def forward(self, src, tgt):
        #         # Embed the input and target
        #         src_embed = self.embedding(src)
        #         tgt_embed = self.embedding(tgt)
        #
        #         # Transpose to match the transformer's input shape
        #         src_embed = src_embed.permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        #         tgt_embed = tgt_embed.permute(1, 0, 2)
        #
        #         # Pass through the transformer
        #         transformer_out = self.transformer(src_embed, tgt_embed)
        #
        #         # Final output layer
        #         output = self.fc_out(transformer_out)
        #         return output.permute(1, 0, 2)  # Back to (batch, seq_len, input_dim): (32, 24, 5) match the transformer's input shape
        src_embed = src_embed.permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        tgt_embed = tgt_embed.permute(1, 0, 2)

        # Pass through the transformer
        transformer_out = self.transformer(src_embed, tgt_embed)

        # Final output layer
        output = self.fc_out(transformer_out)
        return output.permute(1, 0, 2)  # Back to (batch, seq_len, input_dim): (32, 24, 5)


