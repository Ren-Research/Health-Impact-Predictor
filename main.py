import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models.fuel_mix_predictor import TimeSeriesTransformer
from models.dispersion_dnn import HealthImpactMLP
from pipeline_and_loss import TransformerWithMLP, CustomLoss

# ===============================
# Data Preparation
# ===============================
def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)

    # Convert percentages to float
    percentage_cols = [
        "Coal_percentage", "Natural_Gas_percentage",
        "Oil_percentage", "Nuclear_percentage", "Renewable_percentage"
    ]
    for col in percentage_cols:
        df[col] = df[col].str.rstrip('%').astype(float) / 100.0

    # Normalize features
    df[percentage_cols] = (df[percentage_cols] - df[percentage_cols].mean()) / df[percentage_cols].std()

    return df

class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_features, target_features, input_steps, output_steps):
        self.data = data
        self.input_features = input_features
        self.target_features = target_features
        self.input_steps = input_steps
        self.output_steps = output_steps

    def __len__(self):
        return len(self.data) - self.input_steps - self.output_steps + 1

    def __getitem__(self, idx):
        x = self.data.iloc[idx: idx + self.input_steps][self.input_features].values
        y = self.data.iloc[idx + self.input_steps: idx + self.input_steps + self.output_steps][
            self.target_features
        ].values
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ===============================
# Model Initialization
# ===============================
def initialize_model():
    """Initialize the Transformer and MLP models and combine them."""
    transformer = TimeSeriesTransformer(input_dim=5, embed_dim=64, num_heads=4)
    mlp = HealthImpactMLP(input_dim=5, hidden_dim=128, output_dim=2)
    combined_model = TransformerWithMLP(transformer, mlp)
    return combined_model

# ===============================
# Training Loop
# ===============================
def train_model(model, dataloader, loss_fn, optimizer, num_epochs):
    """Train the combined model."""
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            src, targets = batch
            fuel_mix_target = targets[:, :, :5]  # Assuming the first 5 columns are fuel mix targets
            health_targets = targets[:, :, 5:]  # Assuming the next 2 columns are health cost targets

            # Forward pass
            fuel_mix_pred, health_pred = model(src, fuel_mix_target)

            # Compute loss
            loss = loss_fn(
                fuel_mix_pred=fuel_mix_pred,
                fuel_mix_target=fuel_mix_target,
                health_pred=health_pred,
                health_internal_target=health_targets[:, :, 0],
                health_external_target=health_targets[:, :, 1]
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

# === Evaluation Function ===
def evaluate_model(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_loss:.4f}")
    return avg_loss

# ===============================
# Main Execution
# ===============================
def main():
    # Configuration
    file_path = "./CISO_dataset.csv"
    fulemix_features = [
        "Coal_percentage", "Natural_Gas_percentage",
        "Oil_percentage", "Nuclear_percentage", "Renewable_percentage"
    ]
    # target_features = ["internal_health_cost", "external_health_cost"]
    health_features = ["internal_health_cost", "external_health_cost"]
    total_features = ["Coal_percentage", "Natural_Gas_percentage", "Oil_percentage", "Nuclear_percentage", "Renewable_percentage",
                      "internal_health_cost", "external_health_cost"]
    input_steps = 24
    output_steps = 24
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    beta = 0.2

    # Load and preprocess data
    data = load_and_preprocess_data(file_path)

    # Create dataset and dataloaders
    dataset = TimeSeriesDataset(data, fulemix_features, total_features, input_steps, output_steps)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    # model = initialize_model(input_dim=len(input_features), embed_dim=54, num_heads=4, mlp_hidden_dim=128,
    #                          output_dim=len(target_features))
    model = initialize_model()
    loss_fn = CustomLoss(beta=beta)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    print("Starting training...")
    train_model(model, train_loader, loss_fn, optimizer, num_epochs)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, val_loader, loss_fn)

if __name__ == "__main__":
    main()
