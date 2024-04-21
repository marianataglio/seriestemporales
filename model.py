from imports import * 

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size,1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
        
def train_model(model, loader, optimizer, loss_fn, n_epochs=100, print_interval=10, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    train_loss = []
    train_rmse = []

    for epoch in range(n_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_rmse = 0.0
        num_samples = 0
        for X_batch, y_batch in loader:
            
            # Forward pass
            y_pred = model(X_batch)

            # Loss
            loss = loss_fn(y_pred, y_batch)
            epoch_train_loss += loss.item()

            # RMSE
            rmse = torch.sqrt(loss)
            epoch_train_rmse += rmse.item() ** 2
            num_samples += len(X_batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        # Calculate RMSE for each epoch
        epoch_train_rmse = np.sqrt(epoch_train_rmse / num_samples)

        #if (epoch + 1) % print_interval == 0:
           # print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {epoch_train_loss:.4f},  Train RMSE: {epoch_train_rmse:.4f}")

        train_loss.append(epoch_train_loss)
        train_rmse.append(epoch_train_rmse)
    
    return train_loss, train_rmse

import torch
def evaluate_model_last_prediction(model, X_train, y_train, X_test, y_test, scaler_y, loss_fn):
    train_preds = []
    train_loss = []
    test_loss = []
    test_rmse = []
    train_rmse = []
    test_preds = []

    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        epoch_test_loss = 0.0
        epoch_test_rmse = 0.0 

        # Predictions and evaluation on training set
        y_pred_train = model(X_train)
        # Keep the last prediction of the lookback window
        y_pred_train_unscaled = scaler_y.inverse_transform(y_pred_train[:, -1, :].detach().numpy().reshape(-1, 1))
        y_train_unscaled = scaler_y.inverse_transform(y_train[:, -1, :].detach().numpy().reshape(-1, 1))
        y_train_unscaled = torch.tensor(y_train_unscaled)
        y_pred_train_unscaled = torch.tensor(y_pred_train_unscaled)
        
        train_preds.append(y_pred_train_unscaled)
        train_loss.append(loss_fn(y_pred_train, y_train).item())
        train_rmse.append(np.sqrt(loss_fn(y_pred_train_unscaled, y_train_unscaled).item()))

        # Predictions and evaluation on test set
        y_pred_test = model(X_test)
        # Keep the last prediction of the lookback window
        y_pred_test_unscaled = scaler_y.inverse_transform(y_pred_test[:, -1, :].detach().numpy().reshape(-1, 1))
        y_test_unscaled = scaler_y.inverse_transform(y_test[:, -1, :].detach().numpy().reshape(-1, 1))
        y_test_unscaled = torch.tensor(y_test_unscaled)
        y_pred_test_unscaled = torch.tensor(y_pred_test_unscaled)

        
        test_preds.append(y_pred_test_unscaled)
        test_loss.append(loss_fn(y_pred_test, y_test).item()) 
        test_rmse.append(np.sqrt(loss_fn(y_pred_test_unscaled, y_test_unscaled).item()))

    return train_preds, train_loss, test_loss, train_rmse, test_rmse, test_preds
