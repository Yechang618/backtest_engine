# /data/cye_temp/workspace/backtest_engine/src/transformer_model.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class SimpleTabularTransformer(nn.Module):
    """轻量级表格 Transformer 回归器"""
    def __init__(self, input_dim, hidden_dim=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        # 将每个特征视为一个 token，投影到 hidden_dim
        self.input_proj = nn.Linear(1, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*2, 
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: (batch, input_dim) -> (batch, input_dim, 1)
        x = x.unsqueeze(-1)
        x = self.input_proj(x)  # (batch, input_dim, hidden_dim)
        x = self.transformer(x) # (batch, input_dim, hidden_dim)
        x = x.mean(dim=1)       # Mean pooling over features: (batch, hidden_dim)
        return self.head(x).squeeze(-1)

class PyTorchTabularRegressor(BaseEstimator, RegressorMixin):
    """Scikit-Learn 兼容的 PyTorch 模型包装器"""
    def __init__(self, input_dim, hidden_dim=128, n_heads=4, n_layers=2, epochs=30, batch_size=512, lr=1e-3, device='cpu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns.tolist()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).to(self.device)
        
        self.model = SimpleTabularTransformer(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim, 
            n_heads=self.n_heads, n_layers=self.n_layers
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        return preds