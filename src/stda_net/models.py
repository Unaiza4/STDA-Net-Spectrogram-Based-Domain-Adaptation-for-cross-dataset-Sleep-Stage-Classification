import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        out = self.drop(self.act(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class CNNEncoder(nn.Module):
    def __init__(self, feature_dim=128, dropout=0.3):
        super().__init__()
        self.c1 = ConvBlock(1, 32, stride=1, dropout=dropout)
        self.r1 = ResBlock(32, dropout=dropout)
        self.c2 = ConvBlock(32, 64, stride=2, dropout=dropout)
        self.r2 = ResBlock(64, dropout=dropout)
        self.c3 = ConvBlock(64, 128, stride=2, dropout=dropout)
        self.r3 = ResBlock(128, dropout=dropout)
        self.c4 = ConvBlock(128, 128, stride=1, dropout=dropout)
        self.r4 = ResBlock(128, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_proj = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.r1(self.c1(x))
        x = self.r2(self.c2(x))
        x = self.r3(self.c3(x))
        x = self.r4(self.c4(x))
        return self.feat_proj(self.pool(x).flatten(1))


class CNNOnly(nn.Module):
    def __init__(self, num_classes=5, feature_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = CNNEncoder(feature_dim, dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.classifier(feat)
        return feat, logits


class CNNAux(nn.Module):
    def __init__(self, num_classes=5, feature_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = CNNEncoder(feature_dim, dropout)
        self.aux_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim

    def forward(self, x):
        feat = self.encoder(x)
        main_logits = self.classifier(feat)
        aux_logits = self.aux_head(feat)
        return feat, main_logits, aux_logits


class STDASequenceModel(nn.Module):
    def __init__(self, num_classes=5, feature_dim=128, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = CNNEncoder(feature_dim, dropout)
        self.aux_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.bilstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.main_head = nn.Linear(lstm_hidden * 2, num_classes)
        self.feature_dim = feature_dim

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        epoch_feat = self.encoder(x.view(batch_size * seq_len, channels, height, width))
        aux_logits = self.aux_head(epoch_feat).view(batch_size, seq_len, -1)
        seq_feat = epoch_feat.view(batch_size, seq_len, -1)
        seq_out, _ = self.bilstm(seq_feat)
        main_logits = self.main_head(seq_out)
        return epoch_feat, aux_logits, main_logits


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, alpha=1.0):
        return self.net(GradientReversal.apply(x, alpha))


def compute_da_lambda(epoch, total_epochs):
    p = epoch / max(total_epochs, 1)
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
