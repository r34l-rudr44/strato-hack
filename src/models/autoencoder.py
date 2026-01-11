"""
LSTM Autoencoder Anomaly Detector
=================================

An autoencoder using LSTM layers for detecting anomalies in time series data.
The model learns to reconstruct normal patterns; high reconstruction error
indicates anomalies.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base import BaseAnomalyDetector


class LSTMEncoder(nn.Module):
    """LSTM Encoder network."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        latent = self.fc(h_n[-1])  # (batch, latent_dim)
        
        return latent, (h_n, c_n)


class LSTMDecoder(nn.Module):
    """LSTM Decoder network."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        n_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        self.fc_in = nn.Linear(latent_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (batch, latent_dim)
        
        # Expand latent to sequence
        h = self.fc_in(z).unsqueeze(1)  # (batch, 1, hidden_dim)
        h = h.repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)
        
        # Decode through LSTM
        lstm_out, _ = self.lstm(h)
        
        # Output projection
        out = self.fc_out(lstm_out)  # (batch, seq_len, output_dim)
        
        return out


class LSTMAutoencoder(nn.Module):
    """Complete LSTM Autoencoder."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        seq_len: int = 50,
        n_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=n_layers,
            dropout=dropout
        )
        
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            n_layers=n_layers,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent, _ = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self.encoder(x)
        return latent


class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """
    LSTM Autoencoder based anomaly detector.
    
    Learns to reconstruct normal sequences. Anomalies have
    high reconstruction error.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        seq_len: int = 50,
        n_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
        device: Optional[str] = None,
        name: str = "LSTMAutoencoder"
    ):
        """
        Initialize LSTM Autoencoder detector.
        
        Args:
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            seq_len: Sequence length
            n_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
            patience: Early stopping patience
            device: Compute device ("cuda", "cpu", or None for auto)
            name: Detector name
        """
        super().__init__(name=name)
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model: Optional[LSTMAutoencoder] = None
        self.input_dim: Optional[int] = None
        self.train_losses: list = []
        
    def _create_sequences(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        """Create overlapping sequences from data."""
        sequences = []
        for i in range(len(data) - self.seq_len + 1):
            sequences.append(data[i:i + self.seq_len])
        return np.array(sequences)
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'LSTMAutoencoderDetector':
        """
        Fit the LSTM Autoencoder on training data.
        
        Args:
            X: Can be either:
               - 2D array (n_samples, n_features): Will create sequences
               - 3D array (n_sequences, seq_len, n_features): Already sequenced
            y: Ignored (unsupervised)
        """
        X = np.array(X)
        
        # Handle 2D vs 3D input
        if X.ndim == 2:
            sequences = self._create_sequences(X)
        else:
            sequences = X
            self.seq_len = X.shape[1]
            
        self.input_dim = sequences.shape[2]
        
        # Create model
        self.model = LSTMAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            seq_len=self.seq_len,
            n_layers=self.n_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(sequences).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience_counter = 0
        
        print(f"Training LSTM Autoencoder on {self.device}...")
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                
                reconstructed, _ = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            epoch_loss /= len(dataloader)
            self.train_losses.append(epoch_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {epoch_loss:.6f}")
                
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
        # Compute threshold from reconstruction errors on training data
        self.model.eval()
        with torch.no_grad():
            reconstructed, _ = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2))
            errors = errors.cpu().numpy()
            
        # Set threshold at 95th percentile of training errors
        self.threshold = np.percentile(errors, 95)
        
        self.fitted = True
        print(f"Training complete. Threshold: {self.threshold:.6f}")
        
        return self
        
    def score_samples(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Compute reconstruction error as anomaly score.
        """
        if not self.fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
            
        X = np.array(X)
        
        # Handle 2D vs 3D input
        if X.ndim == 2:
            sequences = self._create_sequences(X)
            # Pad scores to match original length
            pad_start = np.zeros(self.seq_len - 1)
        else:
            sequences = X
            pad_start = np.array([])
            
        X_tensor = torch.FloatTensor(sequences).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed, _ = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2))
            scores = errors.cpu().numpy()
            
        # Pad if we created sequences
        if len(pad_start) > 0:
            scores = np.concatenate([pad_start, scores])
            
        return scores
        
    def get_reconstruction(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Get reconstructed sequences.
        """
        if not self.fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
            
        X = np.array(X)
        
        if X.ndim == 2:
            sequences = self._create_sequences(X)
        else:
            sequences = X
            
        X_tensor = torch.FloatTensor(sequences).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed, _ = self.model(X_tensor)
            
        return reconstructed.cpu().numpy()
        
    def get_latent_representations(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Get latent space representations.
        """
        if not self.fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
            
        X = np.array(X)
        
        if X.ndim == 2:
            sequences = self._create_sequences(X)
        else:
            sequences = X
            
        X_tensor = torch.FloatTensor(sequences).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            latent = self.model.encode(X_tensor)
            
        return latent.cpu().numpy()
        
    def get_params(self) -> Dict[str, Any]:
        """Get detector parameters."""
        return {
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'seq_len': self.seq_len,
            'n_layers': self.n_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }


if __name__ == "__main__":
    # Demo
    import sys
    sys.path.append('..')
    from data.synthetic_generator import generate_demo_dataset
    from sklearn.metrics import classification_report
    
    print("Generating demo data...")
    data, labels = generate_demo_dataset(n_timesteps=2000, anomaly_ratio=0.05)
    
    y_true = labels['is_anomaly'].values
    
    # Split data
    split = int(len(data) * 0.8)
    X_train, X_test = data.iloc[:split], data.iloc[split:]
    y_train, y_test = y_true[:split], y_true[split:]
    
    print("\nTraining LSTM Autoencoder...")
    detector = LSTMAutoencoderDetector(
        hidden_dim=32,
        latent_dim=8,
        seq_len=30,
        n_layers=1,
        epochs=30,
        batch_size=32
    )
    
    detector.fit(X_train.values)
    
    # Need to adjust predictions to match test data length
    scores = detector.score_samples(X_test.values)
    
    # Adjust y_test to match scores length (due to sequence creation)
    y_test_adj = y_test[detector.seq_len - 1:]
    scores_adj = scores[detector.seq_len - 1:]
    
    predictions = (scores_adj > detector.threshold).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test_adj, predictions, target_names=['Normal', 'Anomaly']))
