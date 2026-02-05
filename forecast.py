import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

def generate_multivariate_data(n_steps=3000):
    t = np.arange(n_steps)
    data = np.vstack([
        np.sin(0.02 * t) + np.random.normal(0, 0.1, n_steps),
        np.cos(0.015 * t) + np.random.normal(0, 0.1, n_steps),
        np.sin(0.01 * t) * np.cos(0.02 * t)
    ]).T
    return pd.DataFrame(data, columns=["var1", "var2", "var3"])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=30):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x[:, -1])

def train_model(model, loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return rmse, mae, mape

def arima_forecast(train, test):
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit.forecast(steps=len(test))

def visualize_attention(model, sample):
    model.eval()
    with torch.no_grad():
        emb = model.embedding(sample)
        emb = model.pos_encoder(emb)

        attn_layer = model.transformer.layers[0].self_attn
        _, weights = attn_layer(
            emb, emb, emb, need_weights=True
        )

        plt.imshow(weights[0].cpu(), cmap="viridis")
        plt.colorbar()
        plt.title("Self-Attention Weights")
        plt.show()

def main():
    data = generate_multivariate_data()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    dataset = TimeSeriesDataset(scaled, seq_len=30)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TransformerModel(input_dim=3)
    train_model(model, loader, epochs=10)

    model.eval()
    X, y = next(iter(loader))
    preds = model(X).detach().numpy()
    y_true = y.numpy()

    print("Transformer metrics:", metrics(y_true, preds))

    arima_preds = arima_forecast(scaled[:-100, 0], scaled[-100:, 0])
    print("ARIMA metrics:", metrics(scaled[-100:, 0], arima_preds))

    visualize_attention(model, X[:1])

main()