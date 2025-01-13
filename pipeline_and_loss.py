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
        return loss


import torch
import torch.nn as nn


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

    def forward(self, x):
        """
        Forward pass through the combined model.

        Args:
            x (torch.Tensor): Input time-series data of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Predicted internal and external health impacts of shape (batch_size, seq_length, 2).
        """
        # Pass through the Transformer to predict fuel mix
        transformer_output = self.transformer(x)  # Shape: (batch_size, seq_length, input_dim)

        # Pass the transformer's output through the MLP for health impact prediction
        mlp_output = self.mlp(transformer_output)  # Shape: (batch_size, seq_length, 2)

        return mlp_output


# # Example usage:
# # Define transformer and MLP instances
# transformer = TimeSeriesTransformer(input_dim=5, embed_dim=54,
#                                     num_heads=4)  # Replace with your transformer implementation
# mlp = HealthImpactMLP(input_dim=5, hidden_dim=128, output_dim=2)
#
# # Combine transformer and MLP into a single model
# combined_model = TransformerWithMLP(transformer, mlp)
#
# # Dummy input for testing
# x = torch.randn(32, 24, 5)  # Example input: (batch_size=32, seq_length=24, input_dim=5)
#
# # Forward pass
# output = combined_model(x)  # Output: (batch_size=32, seq_length=24, 2)
# print(output.shape)  # Should print torch.Size([32, 24, 2])
