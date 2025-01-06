import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import from our layers.py
from mcqrnn.layers import set_seed, MCQRNN_Dataset, MCQRNN


class MCQRNN_model:
    """
    Encapsulates the MonotoneMLP in a scikit-learn-like API with fit/predict.
    Uses a custom check loss (Huber-smoothed pinball loss).
    """
    def __init__(
        self,
        m_input_size: int,
        i_input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        quantiles,
        dropout_rate: float = 0.4,
        seed: int = None,
        early_stopping_round: int = 10,
    ):
        """
        Args:
            m_input_size: dimension for the monotonic input (often 1 for a single quantile).
            i_input_size: dimension for non-monotonic inputs X.
            hidden_size1: neurons in the 1st hidden layer.
            hidden_size2: neurons in the 2nd hidden layer.
            quantiles: list of quantile values to model (e.g. [0.1, 0.5, 0.9]).
            dropout_rate: dropout probability.
            seed: random seed for reproducibility.
            early_stopping_round: patience for early stopping.
        """
        if seed is not None:
            set_seed(seed)

        self.model = MCQRNN(
            m_input_size=m_input_size,
            i_input_size=i_input_size,
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            seed=seed,
            dropout_rate=dropout_rate,
        )
        self.scaler_x = StandardScaler()
        self.quantiles = quantiles
        self.early_stopping_round = early_stopping_round

    def _MCQRNN_check_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        quantile: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Huber-smoothed pinball loss for a given quantile.
        """
        errors = targets - predictions
        # Huber part
        huber = (
            (errors.pow(2) / (2 * alpha)) * (errors.abs() <= alpha)
            + (errors.abs() - alpha / 2) * (errors.abs() > alpha)
        )
        # Weighted by (tau) if error >= 0 or (1 - tau) if error < 0
        check_loss = (
            torch.sum((quantile * huber)[errors >= 0])
            + torch.sum(((1 - quantile) * huber)[errors < 0])
        ) / huber.size(0)

        return check_loss

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 500,
        num_epochs: int = 1000,
        l1_lambda: float = 0.0,
        random_state: int = 0,
        verbose: int = 0,
    ):
        """
        Train the model using the provided optimizer, data, and parameters.
        """
        # Scale input data
        X = self.scaler_x.fit_transform(X)
        # Convert list of quantiles to np.ndarray shape [Q, 1]
        q = np.array(self.quantiles).reshape(-1, 1)

        # Train/validation split
        tr_X, val_X, tr_y, val_y = train_test_split(
            X, y, test_size=0.3, random_state=random_state
        )

        # Create Datasets
        train_dataset = MCQRNN_Dataset(tr_X, tr_y, q)
        val_dataset = MCQRNN_Dataset(val_X, val_y, q)

        # Create DataLoader for training
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        train_loss = []
        val_loss = []
        best_val_loss = float("inf")
        best_epoch = 0

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0

            for batch_m, batch_i, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass
                predictions = self.model(batch_m, batch_i)
                # Calculate the custom check loss
                loss = self._MCQRNN_check_loss(predictions, batch_y, batch_m, alpha=0.0001)

                # L1 regularization (optional)
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss = loss + l1_lambda * l1_norm

                # Backprop and update
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_y.size(0)

            average_loss = total_loss / len(train_dataset)
            train_loss.append(average_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(val_dataset.data_m, val_dataset.data_i)
                val_loss_epoch = self._MCQRNN_check_loss(
                    val_predictions, val_dataset.labels, val_dataset.data_m, alpha=0.0001
                )
                val_loss.append(val_loss_epoch)

            # Print progress if requested
            if verbose != 0 and (epoch + 1) % verbose == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {train_loss[-1]:.6f}, "
                    f"Validation Loss: {val_loss[-1]:.6f}"
                )

            # Early stopping
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                best_epoch = epoch
            elif (epoch - best_epoch) >= self.early_stopping_round:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print("Training Finished")

    def predict(self, X: np.ndarray, quantiles=None) -> np.ndarray:
        """
        Predict quantiles for new data X. If no quantiles are specified,
        use the ones from initialization.
        """
        # Scale new data
        X = self.scaler_x.transform(X)

        # Use stored quantiles if none are provided
        if quantiles is None:
            quantiles = self.quantiles
        q = np.array(quantiles).reshape(-1, 1)

        # Create input for model
        data_m = torch.tensor(np.repeat(q, len(X)), dtype=torch.float32)
        data_i = torch.tensor(np.tile(X, (len(quantiles), 1)), dtype=torch.float32)

        # Inference
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(data_m, data_i)

        # Reshape to [N, Q]
        predictions = predictions.view(len(quantiles), -1).T
        return predictions.numpy()
