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

MODELS = [
    "conv", "mlp", "lstm", "cnn-lstm",
    #"trmf",
    "lstnet-skip",
    #"darnn"
    "deepglo", "tft", "deepar", "deepstate", "ar",
    #"decision_tree", "random_forest", "xgboost",
    "mq-cnn", "deepfactor"
]


def gaussian_nll_loss(mean, sigma, target):
    eps = 1e-9
    sigma = torch.clamp(sigma, min=eps)
    loss = 0.5 * torch.log(sigma ** 2 + eps) + 0.5 * ((target - mean) ** 2 / (sigma ** 2 + eps))
    return loss.mean()

def trmf_loss(model, x_batch, y_batch, lambda_f=0.01, lambda_g=0.01, lambda_a=0.01):
    y_pred = model(x_batch)
    if y_pred.dim() == 2 and y_pred.size(-1) > 1:
        mse = F.mse_loss(y_pred, y_batch)
    else:
        mse = F.mse_loss(y_pred.squeeze(-1), y_batch.squeeze(-1))
    reg_f = torch.sum(model.F ** 2)
    reg_g = torch.sum(model.G ** 2)
    reg_a = torch.sum(model.A ** 2)
    return mse + lambda_f * reg_f + lambda_g * reg_g + lambda_a * reg_a

class ConvModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, kernel_size=3, channels=32):
        super(ConvModel, self).__init__()
        self.n_timesteps = n_timesteps
        self.horizon = horizon
        self.conv = nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        conv_out_size = (n_timesteps - kernel_size + 1) * channels
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, horizon)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv(x))
        x = x.flatten(start_dim=1)
        return self.fc(x)

class MLPModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1):
        super(MLPModel, self).__init__()
        self.n_timesteps = n_timesteps
        self.horizon = horizon
        self.fc = nn.Sequential(
            nn.Linear(n_timesteps, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, horizon)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, n_timesteps, hidden_size=100, horizon=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, horizon)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.relu(x)
        return self.fc(x)

class ConvLSTMModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, conv_channels=32, lstm_hidden=50, kernel_size=3):
        super(ConvLSTMModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, horizon)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        return self.fc(x)

class TRMFModel(nn.Module):
    def __init__(self, n_timesteps, rank=5, horizon=1):
        super(TRMFModel, self).__init__()
        self.n_timesteps = n_timesteps
        self.rank = rank
        self.horizon = horizon
        self.F = nn.Parameter(torch.randn(n_timesteps, rank))
        self.G = nn.Parameter(torch.randn(rank, 1))
        self.A = nn.Parameter(torch.eye(rank))
        self.bias = nn.Parameter(torch.zeros(1))
        self.x_linear = nn.Linear(1, 1)
        self.fc_horizon = nn.Linear(1, horizon)
    def forward(self, x):
        f_last = self.F[-1]
        f_next = self.A @ f_last
        latent_pred = f_next @ self.G + self.bias
        x_last = x[:, -1, :]
        x_component = self.x_linear(x_last)
        combined = latent_pred.unsqueeze(0) + x_component
        return self.fc_horizon(combined)

class LSTNetSkipModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, kernel_size=3, conv_channels=50,
                 rnn_hidden_size=100, skip=2, skip_rnn_hidden_size=50, highway_window=3,
                 dropout=0.2):
        super(LSTNetSkipModel, self).__init__()
        self.n_timesteps = n_timesteps
        self.horizon = horizon
        self.kernel_size = kernel_size
        self.conv_channels = conv_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.skip = skip
        self.skip_rnn_hidden_size = skip_rnn_hidden_size
        self.highway_window = highway_window

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=kernel_size)
        self.gru = nn.GRU(input_size=conv_channels, hidden_size=rnn_hidden_size, batch_first=True)
        self.skip_gru = nn.GRU(input_size=conv_channels, hidden_size=skip_rnn_hidden_size, batch_first=True)
        self.pt = (n_timesteps - kernel_size + 1) // skip
        self.fc = nn.Linear(rnn_hidden_size + self.pt * skip_rnn_hidden_size, horizon)
        if highway_window > 0:
            self.highway = nn.Linear(highway_window, horizon)
        else:
            self.highway = None
    def forward(self, x):
        batch_size = x.size(0)
        c = x.permute(0, 2, 1)
        c = self.relu(self.conv(c))
        c = self.dropout(c)
        L = c.size(2)
        rnn_input = c.permute(0, 2, 1)
        rnn_output, _ = self.gru(rnn_input)
        h_main = rnn_output[:, -1, :]
        pt = self.pt
        if pt > 0 and L >= pt * self.skip:
            s = c[:, :, -pt * self.skip:]
            s = s.permute(0, 2, 1)
            s = s.view(batch_size, pt, self.skip, self.conv_channels)
            s = s.reshape(batch_size * pt, self.skip, self.conv_channels)
            skip_rnn_output, _ = self.skip_gru(s)
            s_last = skip_rnn_output[:, -1, :]
            s_last = s_last.view(batch_size, pt * self.skip_rnn_hidden_size)
        else:
            s_last = torch.zeros(batch_size, self.pt * self.skip_rnn_hidden_size, device=x.device)
        combined = torch.cat([h_main, s_last], dim=1)
        res = self.fc(combined)
        if self.highway is not None:
            if self.highway_window > x.size(1):
                hw = x[:, -x.size(1):, :]
            else:
                hw = x[:, -self.highway_window:, :]
            hw = hw.permute(0, 2, 1).squeeze(1)
            hw_out = self.highway(hw)
            res = res + hw_out
        return res

class DARNNModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, encoder_hidden_size=64, decoder_hidden_size=64, attention_dim=32):
        super(DARNNModel, self).__init__()
        self.n_timesteps = n_timesteps
        self.horizon = horizon
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_dim = attention_dim
        self.encoder_gru = nn.GRUCell(input_size=1, hidden_size=encoder_hidden_size)
        self.W_e = nn.Linear(encoder_hidden_size, attention_dim, bias=False)
        self.U_e = nn.Parameter(torch.Tensor(1, attention_dim))
        self.b_e = nn.Parameter(torch.Tensor(1, attention_dim))
        self.v_e = nn.Parameter(torch.Tensor(attention_dim))
        nn.init.xavier_uniform_(self.U_e)
        nn.init.zeros_(self.b_e)
        nn.init.xavier_uniform_(self.W_e.weight)
        nn.init.uniform_(self.v_e, -0.1, 0.1)
        self.fc_init = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.decoder_gru = nn.GRUCell(input_size=1, hidden_size=decoder_hidden_size)
        self.W_d = nn.Linear(decoder_hidden_size, attention_dim, bias=False)
        self.U_d = nn.Linear(encoder_hidden_size, attention_dim, bias=False)
        self.v_d = nn.Parameter(torch.Tensor(attention_dim))
        nn.init.xavier_uniform_(self.W_d.weight)
        nn.init.xavier_uniform_(self.U_d.weight)
        nn.init.uniform_(self.v_d, -0.1, 0.1)
        self.fc_out = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)
        self.y0 = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        batch_size, T, _ = x.size()
        device = x.device
        h = torch.zeros(batch_size, self.encoder_hidden_size, device=device)
        encoder_hiddens = []
        for t in range(T):
            x_t = x[:, t, :]
            h_proj = self.W_e(h)
            h_proj_expanded = h_proj.unsqueeze(1)
            x_proj = x_t.unsqueeze(-1) * self.U_e.unsqueeze(0)
            attn_input = h_proj_expanded + x_proj + self.b_e.unsqueeze(0)
            attn_scores = torch.tanh(attn_input)
            scores = torch.sum(attn_scores * self.v_e, dim=2)
            alpha = F.softmax(scores, dim=1)
            x_t_weighted = alpha * x_t
            h = self.encoder_gru(x_t_weighted, h)
            encoder_hiddens.append(h)
        H = torch.stack(encoder_hiddens, dim=1)
        h_T = h
        d = self.fc_init(h_T)
        outputs = []
        dec_input = self.y0.expand(batch_size, 1)
        for _ in range(self.horizon):
            d = self.decoder_gru(dec_input, d)
            d_proj = self.W_d(d).unsqueeze(1).expand(-1, T, -1)
            H_proj = self.U_d(H)
            attn_temp = torch.tanh(d_proj + H_proj)
            temp_scores = torch.sum(attn_temp * self.v_d, dim=2)
            beta = F.softmax(temp_scores, dim=1).unsqueeze(2)
            context = torch.sum(beta * H, dim=1)
            dec_concat = torch.cat([d, context], dim=1)
            out_t = self.fc_out(dec_concat)
            outputs.append(out_t)
            dec_input = out_t
        outputs = torch.cat(outputs, dim=1)
        return outputs

class DeepGloModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, global_hidden_size=64, local_hidden_size=32, local_window=3, dropout=0.2):
        super(DeepGloModel, self).__init__()
        if local_window > n_timesteps:
            raise ValueError("local_window must be <= n_timesteps")
        self.n_timesteps = n_timesteps
        self.horizon = horizon
        self.local_window = local_window
        self.global_lstm = nn.LSTM(input_size=1, hidden_size=global_hidden_size, batch_first=True)
        self.global_fc = nn.Linear(global_hidden_size, horizon)
        self.local_fc1 = nn.Linear(local_window, local_hidden_size)
        self.local_fc2 = nn.Linear(local_hidden_size, horizon)
        self.fusion_fc = nn.Linear(horizon * 2, horizon)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        batch_size = x.size(0)
        global_out, (h_n, _) = self.global_lstm(x)
        h_global = h_n[-1]
        global_pred = self.global_fc(h_global)
        local_input = x[:, -self.local_window:, :].squeeze(-1)
        local_hidden = self.relu(self.local_fc1(local_input))
        local_pred = self.local_fc2(local_hidden)
        combined = torch.cat([global_pred, local_pred], dim=1)
        fused = self.fusion_fc(combined)
        fused = self.dropout(fused)
        return fused

class GatedResidualNetwork(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_model, d_model)
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        gating = torch.sigmoid(self.gate(residual))
        return residual + x * gating

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class TFTModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super(TFTModel, self).__init__()
        self.n_timesteps = n_timesteps
        self.horizon = horizon
        self.d_model = d_model
        self.input_projection = nn.Linear(1, d_model)
        self.grn = GatedResidualNetwork(d_model, dropout=dropout)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=n_timesteps)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, horizon)
    def forward(self, x):
        x = self.input_projection(x)
        x = self.grn(x)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x[-1]
        return self.fc_out(x)

class DeepARModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, hidden_size=64, num_layers=1, dropout=0.1):
        super(DeepARModel, self).__init__()
        self.n_timesteps = n_timesteps
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, 2)  # outputs mean and log_sigma
    def forward(self, x):
        batch_size = x.size(0)
        out, (h, c) = self.lstm(x)
        means = []
        sigmas = []
        last_input = x[:, -1, :]
        for _ in range(self.horizon):
            out_step, (h, c) = self.lstm(last_input.unsqueeze(1), (h, c))
            out_step = self.dropout(out_step[:, -1, :])
            params = self.fc_out(out_step)
            mean, log_sigma = params.split(1, dim=1)
            sigma = torch.exp(log_sigma)
            means.append(mean)
            sigmas.append(sigma)
            last_input = mean
        means = torch.cat(means, dim=1)
        sigmas = torch.cat(sigmas, dim=1)
        return means, sigmas

class DeepStateModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1, hidden_size=64, num_layers=1, dropout=0.1):
        super(DeepStateModel, self).__init__()
        self.n_timesteps = n_timesteps
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.state_rnn = nn.GRU(1, hidden_size, num_layers=num_layers,
                                batch_first=True, dropout=dropout)
        self.emission = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out, h = self.state_rnn(x)
        last_state = out[:, -1, :]
        last_state = self.dropout(last_state)
        if self.horizon == 1:
            return self.emission(last_state)
        else:
            h_state = h[-1]
            outputs = []
            last_input = x[:, -1, :]
            gru_cell = nn.GRUCell(input_size=1, hidden_size=self.hidden_size).to(x.device)
            for name, param in self.state_rnn.named_parameters():
                if 'weight_ih_l0' in name:
                    gru_cell.weight_ih.data.copy_(param.data)
                if 'weight_hh_l0' in name:
                    gru_cell.weight_hh.data.copy_(param.data)
                if 'bias_ih_l0' in name:
                    gru_cell.bias_ih.data.copy_(param.data)
                if 'bias_hh_l0' in name:
                    gru_cell.bias_hh.data.copy_(param.data)
            for _ in range(self.horizon):
                h_state = gru_cell(last_input, h_state)
                h_state_drop = self.dropout(h_state)
                y_t = self.emission(h_state_drop)
                outputs.append(y_t)
                last_input = y_t
            return torch.cat(outputs, dim=1)

class ARModel(nn.Module):
    def __init__(self, n_timesteps, horizon=1):
        super(ARModel, self).__init__()
        self.n_timesteps = n_timesteps
        self.horizon = horizon
        self.coeffs = nn.Parameter(torch.randn(n_timesteps))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        x = x.squeeze(-1)
        if self.horizon == 1:
            return (x * self.coeffs).sum(dim=1, keepdim=True) + self.bias
        else:
            window = x
            forecasts = []
            for _ in range(self.horizon):
                yhat = (window * self.coeffs).sum(dim=1, keepdim=True) + self.bias
                forecasts.append(yhat)
                window = torch.cat([window[:, 1:], yhat], dim=1)
            return torch.cat(forecasts, dim=1)


class GluonTS_MQCNN_Single(nn.Module):
    def __init__(self, n_timesteps, horizon=1, channels=32, kernel_size=3, dropout=0.1):
        """
        Modified GluonTS-style MQ-CNN model that outputs a single point forecast.
        """
        super(GluonTS_MQCNN_Single, self).__init__()
        self.n_timesteps = n_timesteps
        self.horizon = horizon
        padding = kernel_size // 2  # preserve sequence length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * n_timesteps, horizon)
    def forward(self, x):
        # x: (batch, n_timesteps, 1)
        x = x.permute(0, 2, 1)  # -> (batch, 1, n_timesteps)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.flatten(start_dim=1)
        out = self.fc(x)  # (batch, horizon)
        return out

class GluonTS_DeepFactor(nn.Module):
    def __init__(self, n_timesteps, horizon=1, num_factors=5, rnn_hidden=64, fc_hidden=32):
        super(GluonTS_DeepFactor, self).__init__()
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


class ModelBuilder:
    def __init__(self, model_type="lstm", n_timesteps=WINDOW_SIZE, horizon=HORIZON, rank=5,
                 rf_n_estimators=100, rf_max_depth=None,
                 decision_tree_max_depth=None, xgb_max_depth=None):
        self.model_type = model_type
        self.n_timesteps = n_timesteps
        self.horizon = horizon
        self.rank = rank
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.decision_tree_max_depth = decision_tree_max_depth
        self.xgb_max_depth = xgb_max_depth

    def build_model(self):
        model_factories = {
            "conv": lambda: ConvModel(self.n_timesteps, horizon=self.horizon),
            "mlp": lambda: MLPModel(self.n_timesteps, horizon=self.horizon),
            "lstm": lambda: LSTMModel(self.n_timesteps, horizon=self.horizon),
            "cnn-lstm": lambda: ConvLSTMModel(self.n_timesteps, horizon=self.horizon),
            "trmf": lambda: TRMFModel(self.n_timesteps, rank=self.rank, horizon=self.horizon),
            "lstnet-skip": lambda: LSTNetSkipModel(self.n_timesteps, horizon=self.horizon),
            "darnn": lambda: DARNNModel(self.n_timesteps, horizon=self.horizon),
            "deepglo": lambda: DeepGloModel(self.n_timesteps, horizon=self.horizon),
            "tft": lambda: TFTModel(self.n_timesteps, horizon=self.horizon),
            "deepar": lambda: DeepARModel(self.n_timesteps, horizon=self.horizon),
            "deepstate": lambda: DeepStateModel(self.n_timesteps, horizon=self.horizon),
            "ar": lambda: ARModel(self.n_timesteps, horizon=self.horizon),
            "decision_tree": lambda: DecisionTreeRegressor(max_depth=self.decision_tree_max_depth),
            "random_forest": lambda: RandomForestRegressor(n_estimators=self.rf_n_estimators,
                                                           max_depth=self.rf_max_depth),
            "xgboost": lambda: XGBRegressor(max_depth=self.xgb_max_depth),
            "mq-cnn": lambda: GluonTS_MQCNN_Single(self.n_timesteps, horizon=self.horizon),
            "deepfactor": lambda: GluonTS_DeepFactor(self.n_timesteps, horizon=self.horizon)
        }
        try:
            return model_factories[self.model_type]()
        except KeyError:
            raise ValueError(f"Invalid model type '{self.model_type}'!")

    @staticmethod
    def get_available_models():
        return MODELS
