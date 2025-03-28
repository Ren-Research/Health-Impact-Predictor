import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, beta=0.5):
        super(CustomLoss, self).__init__()
        self.beta = beta

    def forward(self, fuel_mix_pred, fuel_mix_target, health_pred, health_internal_target, health_external_target):
        # Fuel mix prediction loss
        fuel_mix_loss = torch.mean((fuel_mix_pred - fuel_mix_target) ** 2)

        # Internal health impact loss
        internal_loss = torch.mean((health_pred[:, :, 0] - health_internal_target) ** 2)

        # External health impact loss
        external_loss = torch.mean((health_pred[:, :, 1] - health_external_target) ** 2)

        # Combined loss
        loss = self.beta * fuel_mix_loss + (1 - self.beta) / 2 * (internal_loss + external_loss)
        return loss, fuel_mix_loss, internal_loss, external_loss

class TransformerWithMLP(nn.Module):
    def __init__(self, transformer, mlp):
        """
        Initialize the TransformerWithMLP model.

        Args:
            transformer (nn.Module): The time-series transformer model.
            mlp (nn.Module): The MLP model for health impact prediction.
        """
        super(TransformerWithMLP, self).__init__()
        self.transformer = transformer
        self.mlp = mlp

    def forward(self, src, tgt):
        """
        Forward pass through the combined model.

        Args:
            src (torch.Tensor): Source input for the transformer of shape (batch, src_seq_len, input_dim).
            tgt (torch.Tensor): Target input for the transformer of shape (batch, tgt_seq_len, input_dim).

        Returns:
            torch.Tensor: Final output from the MLP of shape (batch, seq_len, output_dim).
        """
        # Pass through the Transformer to predict fuel mix
        transformer_output = self.transformer(src, tgt)  # Shape: (batch_size, seq_length, input_dim)

        # Pass the transformer's output through the MLP for health impact prediction
        mlp_output = self.mlp(transformer_output)  # Shape: (batch_size, seq_length, 2)

        return transformer_output, mlp_output


