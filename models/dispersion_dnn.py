import torch
import torch.nn as nn


class HealthImpactMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super(HealthImpactMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Input x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape

        # Flatten the time dimension for MLP input
        x = x.view(batch_size * seq_len, input_dim)

        # Pass through the MLP
        x = self.model(x)

        # Reshape back to (batch_size, seq_len, output_dim)
        return x.view(batch_size, seq_len, -1)


# # Hyperparameters
# input_dim = 5  # Number of fuel mix features
# hidden_dim = 128  # Hidden layer size
# output_dim = 2  # Internal and external health costs
#
# # Initialize the MLP
# mlp = HealthImpactMLP(input_dim, hidden_dim, output_dim)
#
# # Example input: Transformer output predictions
# # Shape: (batch_size, seq_len, input_dim)
# transformer_output = torch.rand(32, 24, input_dim)
#
# # Predict health impacts
# health_impact = mlp(transformer_output)
#
# # Output shape: (batch_size, seq_len, output_dim)
# print("Health Impact Output Shape:", health_impact.shape)
