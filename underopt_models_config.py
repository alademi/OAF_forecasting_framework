import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

WINDOW_SIZE = 10
HORIZON = 1
LR = 0.001
EPOCHS = 10
BATCH_SIZE = 32

UNDER_OPT_MODELS = [
    "under_opt_conv", "under_opt_mlp", "under_opt_lstm", "under_opt_cnn_lstm",
  "under_opt_lstnet_skip", "under_opt_deepglo",
    "under_opt_tft", "under_opt_deepar", "under_opt_deepstate", "under_opt_ar",
    "under_opt_mq_cnn", "under_opt_deepfactor"
]


class UnderOptConvModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, kernel_size=3, channels=8):
        super(UnderOptConvModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=kernel_size)
        self.tanh = nn.Tanh()
        conv_out_size = (n_timesteps - kernel_size + 1) * channels
        self.fc = nn.Linear(conv_out_size, horizon)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tanh(self.conv(x))
        x = x.flatten(start_dim=1)
        return self.fc(x)


class UnderOptMLPModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1):
        super(UnderOptMLPModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_timesteps, 16),
            nn.Tanh(),
            nn.Linear(16, horizon)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class UnderOptLSTMModel(nn.Module):
    def __init__(self, n_timesteps, hidden_size=32, horizon=1):
        super(UnderOptLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        return self.fc(x)


class UnderOptCNNLSTMModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, conv_channels=8, lstm_hidden=16, kernel_size=3):
        super(UnderOptCNNLSTMModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=kernel_size)
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, horizon)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        return self.fc(x)


class UnderOptDecisionTree:
    def __init__(self, max_depth=2):
        self.model = DecisionTreeRegressor(max_depth=max_depth)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class UnderOptRandomForest:
    def __init__(self, n_estimators=5, max_depth=2):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class UnderOptXGBoost:
    def __init__(self, max_depth=2):
        self.model = XGBRegressor(max_depth=max_depth, n_estimators=5)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class UnderOptLSTNetSkipModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, kernel_size=3, conv_channels=16, rnn_hidden_size=32):
        super(UnderOptLSTNetSkipModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=kernel_size)
        self.gru = nn.GRU(input_size=conv_channels, hidden_size=rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, horizon)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        rnn_out, _ = self.gru(x)
        return self.fc(rnn_out[:, -1, :])


class UnderOptDeepGloModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, hidden_size=16):
        super(UnderOptDeepGloModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class UnderOptTFTModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, d_model=16, n_heads=2, num_layers=1):
        super(UnderOptTFTModel, self).__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.input_proj(x).permute(1, 0, 2)
        x = self.transformer_encoder(x)[-1]
        return self.fc_out(x)


class UnderOptDeepARModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, hidden_size=16):
        super(UnderOptDeepARModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class UnderOptDeepStateModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, hidden_size=16):
        super(UnderOptDeepStateModel, self).__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class UnderOptARModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1):
        super(UnderOptARModel, self).__init__()
        self.coeffs = nn.Parameter(torch.randn(n_timesteps) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return (x.squeeze(-1) * self.coeffs).sum(dim=1, keepdim=True) + self.bias


class UnderOptMQCNN(nn.Module):
    def __init__(self, n_timesteps, horizon=1, channels=8, kernel_size=3):
        super(UnderOptMQCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()

        conv_out_size = (n_timesteps - (kernel_size - 1)) * channels
        self.fc = nn.Linear(conv_out_size, horizon)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch, channels, seq_len)
        x = self.relu(self.conv(x))  # Apply convolution + activation
        x = x.flatten(start_dim=1)  # Flatten before passing to FC layer
        return self.fc(x)


class UnderOptDeepFactor(nn.Module):
    def __init__(self, n_timesteps, horizon=1, num_factors=3, rnn_hidden=32, fc_hidden=16):
        super(UnderOptDeepFactor, self).__init__()
        self.n_timesteps = n_timesteps
        self.horizon = horizon
        self.num_factors = num_factors
        self.rnn = nn.GRU(input_size=1, hidden_size=rnn_hidden, batch_first=True)
        self.fc_mixing = nn.Linear(rnn_hidden, num_factors * horizon)
        self.factor_forecast = nn.Sequential(
            nn.Linear(n_timesteps, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, horizon)
        )
        self.global_factors = nn.Parameter(torch.randn(num_factors, n_timesteps))

    def forward(self, x):
        batch_size = x.size(0)
        _, h = self.rnn(x)
        h = h.squeeze(0)
        mixing = self.fc_mixing(h)
        mixing = mixing.view(batch_size, self.horizon, self.num_factors)
        factor_forecasts = []
        for i in range(self.num_factors):
            factor_i = self.global_factors[i].unsqueeze(0)
            forecast_i = self.factor_forecast(factor_i)
            factor_forecasts.append(forecast_i)
        factor_forecasts = torch.cat(factor_forecasts, dim=0)
        factor_forecasts = factor_forecasts.transpose(0, 1)
        forecasts = []
        for t in range(self.horizon):
            f_t = (mixing[:, t, :] * factor_forecasts[t]).sum(dim=1, keepdim=True)
            forecasts.append(f_t)
        forecasts = torch.cat(forecasts, dim=1)
        return forecasts


class UnderOptModelBuilder:
    def __init__(self, model_type="under_opt_lstm", n_timesteps=WINDOW_SIZE, horizon=HORIZON):
        self.model_type = model_type
        self.n_timesteps = n_timesteps
        self.horizon = horizon

    def build_model(self):
        model_factories = {
            "under_opt_conv": lambda: UnderOptConvModel(self.n_timesteps, horizon=self.horizon),
            "under_opt_mlp": lambda: UnderOptMLPModel(self.n_timesteps, horizon=self.horizon),
            "under_opt_lstm": lambda: UnderOptLSTMModel(self.n_timesteps, horizon=self.horizon),
            "under_opt_cnn_lstm": lambda: UnderOptCNNLSTMModel(self.n_timesteps, horizon=self.horizon),
            "under_opt_lstnet_skip": lambda: UnderOptLSTNetSkipModel(self.n_timesteps, horizon=self.horizon),
            "under_opt_deepglo": lambda: UnderOptDeepGloModel(self.n_timesteps, horizon=self.horizon),
            "under_opt_tft": lambda: UnderOptTFTModel(self.n_timesteps, horizon=self.horizon),
            "under_opt_deepar": lambda: UnderOptDeepARModel(self.n_timesteps, horizon=self.horizon),
            "under_opt_deepstate": lambda: UnderOptDeepStateModel(self.n_timesteps, horizon=self.horizon),
            "under_opt_ar": lambda: UnderOptARModel(self.n_timesteps, horizon=self.horizon),
            "under_opt_mq_cnn": lambda: UnderOptMQCNN(self.n_timesteps, horizon=self.horizon),
            "under_opt_deepfactor": lambda: UnderOptDeepFactor(self.n_timesteps, horizon=self.horizon)
        }
        try:
            return model_factories[self.model_type]()
        except KeyError:
            raise ValueError(f"Invalid model type '{self.model_type}'!")

    @staticmethod
    def get_available_models():
        return UNDER_OPT_MODELS
