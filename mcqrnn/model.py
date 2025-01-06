import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import from our layers.py
from mcqrnn.layers import set_seed, MCQRNN
from mcqrnn.dataset import MCQRNN_Dataset
from mcqrnn.loss import composite_check_loss


class MCQRNN_model:
    def __init__(
        self,
        m_input_size,
        i_input_size,
        hidden_size,
        quantiles,
        dropout_rate,
        seed=None,
        early_stopping_round=10,
    ):
        """
        Args:
            m_input_size: dimension for the monotonic input (often 1 for a single quantile).
            i_input_size: dimension for non-monotonic inputs X.
            hidden_size1: neurons in the hidden layer.
            quantiles: list of quantile values to model (e.g. [0.1, 0.5, 0.9]).
            dropout_rate: dropout probability.
            seed: random seed for reproducibility.
            early_stopping_round: patience for early stopping.
        """
        if seed is not None:
            set_seed(seed)
        self.model = MCQRNN(
            m_input_size,
            i_input_size,
            hidden_size,
            seed=seed,
            dropout_rate=dropout_rate,
        )
        self.scaler_x = StandardScaler()
        self.quantiles = quantiles
        self.early_stopping_round = early_stopping_round

    def fit(
        self,
        X,
        y,
        optimizer,
        batch_size=500,
        num_epochs=1000,
        l1_lambda=0,
        random_state=0,
        verbose=0,
    ):
        # 데이터 전처리 및 분할
        X = self.scaler_x.fit_transform(X)
        q = np.array(self.quantiles).reshape(-1, 1)
        tr_X, val_X, tr_y, val_y = train_test_split(
            X, y, test_size=0.3, random_state=random_state
        )

        # 데이터셋 및 데이터로더 생성
        train_dataset = MCQRNN_Dataset(tr_X, tr_y, q)
        val_dataset = MCQRNN_Dataset(val_X, val_y, q)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 학습 초기화
        train_loss = []
        val_loss = []
        best_val_loss = float("inf")
        best_epoch = 0

        # 모델 학습
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            for batch_x_m, batch_x_i, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.model(batch_x_m, batch_x_i)
                loss = composite_check_loss(
                    predictions, batch_y, batch_x_m, alpha=0.0001
                )
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss = loss + l1_lambda * l1_norm
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_y.size(0)
            average_loss = total_loss / len(train_dataset)
            train_loss.append(average_loss)
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(val_dataset.data_m, val_dataset.data_i)
                val_loss_epoch = composite_check_loss(
                    val_predictions,
                    val_dataset.labels,
                    val_dataset.data_m,
                    alpha=0.0001,
                )
                val_loss.append(val_loss_epoch)
            if verbose != 0:
                if (epoch + 1) % verbose == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss[-1]}, Validation Loss: {val_loss[-1]}"
                    )

            # Early stopping 조건 확인
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                best_epoch = epoch

            if self.early_stopping_round > 0:
                if epoch - best_epoch >= self.early_stopping_round:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print("Training Finished")

    def predict(self, X, quantiles=None):
        X = self.scaler_x.transform(X)

        if quantiles is None:
            quantiles = self.quantiles
        q = np.array(quantiles).reshape(-1, 1)
        data_m = torch.tensor(np.repeat(np.array(q), len(X)), dtype=torch.float32)
        data_i = torch.tensor(np.tile(X, (len(quantiles), 1)), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(data_m, data_i)
        predictions = predictions.view(len(quantiles), -1).T
        return predictions.numpy()
