# -*-coding:utf-8-*-
import torch.nn as nn
import torch
from toolkit.distort import *
import random
from model.frequency import time_to_timefreq, time_to_timefreq_v2
from model.PatchTSMixerLayer import PatchMixerEncoder
from model.layers import TransformerEncoder

seed = 42
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

torch.backends.cudnn.benchmark=True

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

class GatedMultimodalLayer(nn.Module):
    def __init__(self, size_in1, size_in2, size_out=16):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden_sigmoid = nn.Linear(size_out * 2, 1, bias=False)
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden2(x2))
        x = torch.cat((h1, h2), dim=-1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))
        return z * h1 + (1 - z) * h2


class DetectionNet(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, n_fft=4):
        super().__init__()
        self.patch_size = patch_size
        self.n_fft = n_fft
        self.proj_size = patch_size * 2 * (self.n_fft // 2 + 1)
        self.hidden_size = int(expansion_ratio * self.proj_size)
        self.proj = nn.Linear(self.proj_size, self.hidden_size)
        self.act = nn.SELU()
        self.encoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.classifier = nn.Linear(self.hidden_size * 2, 1)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self,
                x: torch.Tensor):
        """
        :param x: batch_size, seq_length
        :return:
        """
        # (batch_size * num_channels, seq_len, 2 * (self.n_fft // 2 + 1))
        batch_size, seq_length = x.shape
        x_freq = time_to_timefreq(x, n_fft=self.n_fft, C=1)
        x_freq = x_freq[:, :seq_length, :]

        # batch_size * num_channels, patch_num, patch_length * fft_length
        patch_freq = x_freq.view(batch_size, seq_length // self.patch_size, -1)

        # batch_size * num_channels, patch_num, hidden_size
        patch_proj = self.proj(patch_freq)
        patch_proj = self.act(patch_proj)
        patch_output, patch_hidden = self.encoder(patch_proj)

        # batch_size * num_channels, patch_num, 1
        classify_result = self.classifier(patch_output)

        patch_anomaly_score = torch.nn.Sigmoid()(classify_result)

        return classify_result, patch_anomaly_score


class DetectionNetTimeFreq(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, n_fft=4):
        super().__init__()
        self.patch_size = patch_size
        self.n_fft = n_fft
        self.proj_size_freq = patch_size * 2 * (self.n_fft // 2 + 1)
        self.proj_size_time = patch_size

        self.hidden_size_freq = expansion_ratio * self.proj_size_freq
        self.hidden_size_time = expansion_ratio * self.proj_size_time

        self.proj_freq = nn.Linear(self.proj_size_freq, self.hidden_size_freq)
        self.proj_time = nn.Linear(self.proj_size_time, self.hidden_size_time)

        self.act = nn.SELU()

        self.encoder_time = nn.GRU(
            input_size=self.hidden_size_time,
            hidden_size=self.hidden_size_time,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        self.encoder_freq = nn.GRU(
            input_size=self.hidden_size_freq,
            hidden_size=self.hidden_size_freq,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        self.classifier_time = nn.Linear(2 * self.hidden_size_time, 1)
        self.classifier_freq = nn.Linear(2 * self.hidden_size_freq, 1)

        nn.init.xavier_uniform_(self.proj_freq.weight)
        nn.init.zeros_(self.proj_freq.bias)
        nn.init.xavier_uniform_(self.proj_time.weight)
        nn.init.zeros_(self.proj_time.bias)

        nn.init.xavier_uniform_(self.classifier_time.weight)
        nn.init.zeros_(self.classifier_time.bias)
        nn.init.xavier_uniform_(self.classifier_freq.weight)
        nn.init.zeros_(self.classifier_freq.bias)

    def forward(self,
                x: torch.Tensor):
        """
        :param x: batch_size, seq_length
        :return:
        """
        # (batch_size * num_channels, seq_len, 2 * (self.n_fft // 2 + 1))
        batch_size, seq_length = x.shape
        x_freq = time_to_timefreq(x, n_fft=self.n_fft, C=1)
        x_freq = x_freq[:, :seq_length, :]
        x_time = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)

        # batch_size * num_channels, patch_num, patch_length * fft_length
        patch_freq = x_freq.view(batch_size, seq_length // self.patch_size, -1)

        # batch_size * num_channels, patch_num, hidden_size
        freq_proj = self.proj_freq(patch_freq)
        freq_proj = self.act(freq_proj)

        time_proj = self.proj_time(x_time)
        time_proj = self.act(time_proj)

        patch_output_freq, patch_hidden_freq = self.encoder_freq(freq_proj)
        patch_output_time, patch_hidden_time = self.encoder_time(time_proj)

        # patch_output_freq = patch_output_freq.reshape(batch_size * (seq_length // self.patch_size), -1)
        # patch_output_time = patch_output_time.reshape(batch_size * (seq_length // self.patch_size), -1)
        #
        # patch_output = self.fusion_net(patch_output_freq, patch_output_time)

        # batch_size * num_channels, patch_num, 1
        classify_result_freq = self.classifier_freq(patch_output_freq)
        classify_result_time = self.classifier_time(patch_output_time)
        classify_result = torch.cat((classify_result_freq, classify_result_time), dim=-1)
        classify_result = torch.max(classify_result, dim=-1, keepdim=True).values

        patch_anomaly_score = torch.nn.Sigmoid()(classify_result)

        return classify_result, patch_anomaly_score


class DetectionNetTime(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, n_fft=4):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = expansion_ratio * patch_size
        self.proj = nn.Linear(self.patch_size, self.hidden_size)
        self.act = nn.SELU()
        self.encoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.classifier = nn.Linear(self.hidden_size * 2, 1)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self,
                x: torch.Tensor):
        """
        :param x: batch_size, seq_length
        :return:
        """
        # (batch_size * num_channels, seq_len, 2 * (self.n_fft // 2 + 1))
        batch_size, seq_length = x.shape
        # batch_size * num_channels, patch_num, patch_length * fft_length
        patch_freq = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)

        # batch_size * num_channels, patch_num, hidden_size
        patch_proj = self.proj(patch_freq)
        patch_proj = self.act(patch_proj)
        patch_output, patch_hidden = self.encoder(patch_proj)

        # batch_size * num_channels, patch_num, 1
        classify_result = self.classifier(patch_output)

        patch_anomaly_score = torch.nn.Sigmoid()(classify_result)

        return classify_result, patch_anomaly_score

class DetectionNetTimeFreqV2(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, n_fft=4, fusion_type='max'):
        super().__init__()
        self.patch_size = patch_size
        self.n_fft = n_fft
        self.proj_size_freq = patch_size * 2 * (self.n_fft // 2 + 1)
        self.proj_size_time = patch_size
        self.hidden_size = expansion_ratio * self.proj_size_freq
        self.proj_freq = nn.Sequential(
            nn.Linear(self.proj_size_freq, self.hidden_size),
            # nn.LayerNorm(self.hidden_size),
            nn.SELU()
        )
        self.proj_time = nn.Sequential(
            nn.Linear(self.proj_size_time, self.hidden_size),
            # nn.LayerNorm(self.hidden_size),
            nn.SELU()
        )
        self.fusion_type = fusion_type
        # self.encoder_time = nn.GRU(
        #     input_size=self.hidden_size,
        #     hidden_size=self.hidden_size,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     bidirectional=True,
        #     dropout=0.1
        # )
        # self.encoder_freq = nn.GRU(
        #     input_size=self.hidden_size,
        #     hidden_size=self.hidden_size,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     bidirectional=True,
        #     dropout=0.1
        # )
        self.encoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        if fusion_type =='gate':
            self.fusion_net = GatedMultimodalLayer(2 * self.hidden_size, 2 * self.hidden_size, 2 * self.hidden_size)

        self.classifier_time = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            # nn.LayerNorm(self.hidden_size),
            nn.SELU(),
            nn.Linear(self.hidden_size, 1, bias=False),
        )
        self.classifier_freq = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            # nn.LayerNorm(self.hidden_size),
            nn.SELU(),
            nn.Linear(self.hidden_size, 1, bias=False),
        )
        self.classifier_residual = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            # nn.LayerNorm(self.hidden_size),
            nn.SELU(),
            nn.Linear(self.hidden_size, 1, bias=False),
        )

        nn.init.xavier_uniform_(self.proj_freq[0].weight)
        nn.init.zeros_(self.proj_freq[0].bias)
        nn.init.xavier_uniform_(self.proj_time[0].weight)
        nn.init.zeros_(self.proj_time[0].bias)

        # 初始化 classifier_time
        for layer in self.classifier_time:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # 初始化 classifier_freq
        for layer in self.classifier_freq:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # 初始化 classifier_residual
        for layer in self.classifier_residual:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self,
                x: torch.Tensor,
                x_refer: torch.Tensor =  None):
        """
        :param x: batch_size, seq_length
        :param x_refer: batch_size, seq_length
        :return:
        """
        if x_refer is not None:
            x = torch.cat((x, x_refer), dim=0) # batch_size * 2, seq_length

        batch_size, seq_length = x.shape
        x_freq = time_to_timefreq(x.clone(), n_fft=self.n_fft, C=1) # (batch_size * num_channels, seq_len, 2 * (self.n_fft // 2 + 1))
        x_freq = x_freq[:, :seq_length, :]
        x_time = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)

        patch_freq = x_freq.view(batch_size, seq_length // self.patch_size, -1)  # batch_size * num_channels, patch_num, patch_length * fft_length
        patch_time = x_time.view(batch_size, seq_length // self.patch_size, -1)  # batch_size * num_channels, patch_num, patch_length

        # batch_size * num_channels, patch_num, hidden_size
        freq_proj = self.proj_freq(patch_freq)
        time_proj = self.proj_time(patch_time)

        all_proj = torch.cat((freq_proj, time_proj), dim=0) # batch_size * 2, patch_num, hidden_size
        patch_output_all, _ = self.encoder(all_proj) # batch_size * 2, patch_num, hidden_size
        patch_output_freq, patch_output_time = torch.split(patch_output_all, batch_size, dim=0) # batch_size, patch_num, hidden_size
        # patch_output_freq, patch_hidden_freq = self.encoder_freq(freq_proj)
        # patch_output_time, patch_hidden_time = self.encoder_time(time_proj)

        classify_residual_result: torch.Tensor = None
        patch_residual_freq: torch.Tensor = None
        patch_residual_time: torch.Tensor = None
        if x_refer is not None:
            patch_output_freq, patch_ref_freq = torch.split(patch_output_freq, batch_size // 2, dim=0)
            patch_output_time, patch_ref_time = torch.split(patch_output_time, batch_size // 2, dim=0)
            patch_residual_freq = patch_output_freq - patch_ref_freq
            patch_residual_time = patch_output_time - patch_ref_time
            classify_result_residual_freq = self.classifier_residual(patch_residual_freq)
            classify_result_residual_time = self.classifier_residual(patch_residual_time)
            classify_residual_result = torch.cat((classify_result_residual_freq, classify_result_residual_time), dim=-1)

        # batch_size * num_channels, patch_num, 1
        classify_result_freq = self.classifier_freq(patch_output_freq)
        classify_result_time = self.classifier_time(patch_output_time)
        classify_result = torch.cat((classify_result_freq, classify_result_time), dim=-1)

        if self.fusion_type =='max':
            classify_result = torch.max(classify_result, dim=-1, keepdim=True).values
            if x_refer is not None:
                classify_residual_result = torch.max(classify_residual_result, dim=-1, keepdim=True).values
        elif self.fusion_type =='mean':
            classify_result = torch.mean(classify_result, dim=-1, keepdim=True)
            if x_refer is not None:
                classify_residual_result = torch.mean(classify_residual_result, dim=-1, keepdim=True)
        elif self.fusion_type == 'gate':
            patch_output_fusion = self.fusion_net(patch_output_freq, patch_output_time)
            classify_result = self.classifier_freq(patch_output_fusion)
            if x_refer is not None:
                patch_residual_fusion = self.fusion_net(patch_residual_freq, patch_residual_time)
                classify_residual_result = self.classifier_residual(patch_residual_fusion)
        elif self.fusion_type == 'add':
            patch_output_fusion = patch_output_freq + patch_output_time
            classify_result = self.classifier_freq(patch_output_fusion)
            if x_refer is not None:
                patch_residual_fusion = patch_residual_freq + patch_residual_time
                classify_residual_result = self.classifier_residual(patch_residual_fusion)
        else:
            raise ValueError('fusion_type must be max, mean, gate or add')

        patch_anomaly_score = torch.nn.Sigmoid()(classify_result)
        if classify_residual_result is not None:
            patch_residual_anomaly_score = torch.nn.Sigmoid()(classify_residual_result)
            return classify_result, patch_anomaly_score, classify_residual_result, patch_residual_anomaly_score

        return classify_result, patch_anomaly_score

class DetectionNetTimeFreqPoint(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, n_fft=4, fusion_type='max'):
        super().__init__()
        self.patch_size = patch_size
        self.n_fft = n_fft
        self.proj_size_freq = patch_size * 2 * (self.n_fft // 2 + 1)
        self.proj_size_time = patch_size
        self.hidden_size = expansion_ratio * self.proj_size_freq
        self.proj_freq = nn.Linear(self.proj_size_freq, self.hidden_size)
        self.proj_time = nn.Linear(self.proj_size_time, self.hidden_size)
        self.fusion_type = fusion_type
        self.act = nn.SELU()
        self.encoder_time = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.encoder_freq = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        if fusion_type =='gate':
            self.fusion_net = GatedMultimodalLayer(2 * self.hidden_size, 2 * self.hidden_size, 2 * self.hidden_size)


        # self.classifier_time = nn.Linear(self.hidden_size * 2, 1)
        # self.classifier_freq = nn.Linear(self.hidden_size * 2, 1)
        self.point_time_proj = nn.Linear(2 * self.hidden_size, patch_size)
        self.point_freq_proj = nn.Linear(2 * self.hidden_size, patch_size)

        # nn.init.xavier_uniform_(self.proj_freq.weight)
        # nn.init.zeros_(self.proj_freq.bias)
        # nn.init.xavier_uniform_(self.proj_time.weight)
        # nn.init.zeros_(self.proj_time.bias)
        # nn.init.xavier_uniform_(self.classifier_time.weight)
        # nn.init.zeros_(self.classifier_time.bias)
        # nn.init.xavier_uniform_(self.classifier_freq.weight)
        # nn.init.zeros_(self.classifier_freq.bias)

    def forward(self,
                x: torch.Tensor):
        """
        :param x: batch_size, seq_length
        :return:
        """
        # (batch_size * num_channels, seq_len, 2 * (self.n_fft // 2 + 1))
        batch_size, seq_length = x.shape
        x_freq = time_to_timefreq(x, n_fft=self.n_fft, C=1)
        x_freq = x_freq[:, :seq_length, :]
        x_time = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)

        # batch_size * num_channels, patch_num, patch_length * fft_length
        patch_freq = x_freq.view(batch_size, seq_length // self.patch_size, -1)
        patch_time = x_time.view(batch_size, seq_length // self.patch_size, -1)

        # batch_size * num_channels, patch_num, hidden_size
        freq_proj = self.proj_freq(patch_freq)
        freq_proj = self.act(freq_proj)

        time_proj = self.proj_time(patch_time)
        time_proj = self.act(time_proj)

        patch_output_freq, patch_hidden_freq = self.encoder_freq(freq_proj)
        patch_output_time, patch_hidden_time = self.encoder_time(time_proj)

        # batch_size * num_channels, patch_num, patch_size
        classify_result_freq = self.point_freq_proj(patch_output_freq).unsqueeze(-1)
        classify_result_time = self.point_time_proj(patch_output_time).unsqueeze(-1)
        classify_result = torch.cat((classify_result_freq, classify_result_time), dim=-1)

        if self.fusion_type =='max':
            classify_result = torch.max(classify_result, dim=-1, keepdim=True).values.squeeze(-1)
        elif self.fusion_type =='mean':
            classify_result = torch.mean(classify_result, dim=-1, keepdim=True)
        elif self.fusion_type == 'gate':
            patch_output_fusion = self.fusion_net(patch_output_freq, patch_output_time)
            classify_result = self.classifier_freq(patch_output_fusion)
        elif self.fusion_type == 'add':
            patch_output_fusion = patch_output_freq + patch_output_time
            classify_result = self.classifier_freq(patch_output_fusion)
        else:
            raise ValueError('fusion_type must be max, mean, gate or add')

        patch_anomaly_score = torch.mean(classify_result, dim=-1, keepdim=True)
        patch_anomaly_score = torch.nn.Sigmoid()(patch_anomaly_score)

        return classify_result, patch_anomaly_score


class DetectionNetTimeFreqWindow(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, n_fft=4, fusion_type='max'):
        super().__init__()
        self.patch_size = patch_size
        self.n_fft = n_fft
        self.proj_size_freq = patch_size * 2 * (self.n_fft // 2 + 1)
        self.proj_size_time = patch_size
        self.hidden_size = expansion_ratio * self.proj_size_freq
        self.proj_freq = nn.Linear(self.proj_size_freq, self.hidden_size)
        self.proj_time = nn.Linear(self.proj_size_time, self.hidden_size)
        self.fusion_type = fusion_type
        self.act = nn.SELU()
        self.encoder_time = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.encoder_freq = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        if fusion_type =='gate':
            self.fusion_net = GatedMultimodalLayer(2 * self.hidden_size, 2 * self.hidden_size, 2 * self.hidden_size)


        self.classifier_time = nn.Linear(self.hidden_size * 2, 1)
        self.classifier_freq = nn.Linear(self.hidden_size * 2, 1)
        # self.point_time_proj = nn.Linear(2 * self.hidden_size, patch_size)
        # self.point_freq_proj = nn.Linear(2 * self.hidden_size, patch_size)

        # nn.init.xavier_uniform_(self.proj_freq.weight)
        # nn.init.zeros_(self.proj_freq.bias)
        # nn.init.xavier_uniform_(self.proj_time.weight)
        # nn.init.zeros_(self.proj_time.bias)
        # nn.init.xavier_uniform_(self.classifier_time.weight)
        # nn.init.zeros_(self.classifier_time.bias)
        # nn.init.xavier_uniform_(self.classifier_freq.weight)
        # nn.init.zeros_(self.classifier_freq.bias)

    def forward(self,
                x: torch.Tensor):
        """
        :param x: batch_size, seq_length
        :return:
        """
        # (batch_size * num_channels, seq_len, 2 * (self.n_fft // 2 + 1))
        batch_size, seq_length = x.shape
        x_freq = time_to_timefreq(x, n_fft=self.n_fft, C=1)
        x_freq = x_freq[:, :seq_length, :]
        x_time = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)

        # batch_size * num_channels, patch_num, patch_length * fft_length
        patch_freq = x_freq.view(batch_size, seq_length // self.patch_size, -1)
        patch_time = x_time.view(batch_size, seq_length // self.patch_size, -1)

        # batch_size * num_channels, patch_num, hidden_size
        freq_proj = self.proj_freq(patch_freq)
        freq_proj = self.act(freq_proj)

        time_proj = self.proj_time(patch_time)
        time_proj = self.act(time_proj)

        patch_output_freq, patch_hidden_freq = self.encoder_freq(freq_proj)
        patch_output_time, patch_hidden_time = self.encoder_time(time_proj)

        # batch_size * num_channels, 1, hidden_size
        patch_output_freq = torch.mean(patch_output_freq, dim=1)
        patch_output_time = torch.mean(patch_output_time, dim=1)

        # batch_size * num_channels, 1, 1
        classify_result_freq = self.classifier_freq(patch_output_freq).unsqueeze(-1)
        classify_result_time = self.classifier_time(patch_output_time).unsqueeze(-1)
        # batch_size * num_channels, 1, 2
        classify_result = torch.cat((classify_result_freq, classify_result_time), dim=-1)

        if self.fusion_type =='max':
            # batch_size * num_channels, 1, 1
            classify_result = torch.max(classify_result, dim=-1, keepdim=True).values
        elif self.fusion_type =='mean':
            classify_result = torch.mean(classify_result, dim=-1, keepdim=True)
        elif self.fusion_type == 'gate':
            patch_output_fusion = self.fusion_net(patch_output_freq, patch_output_time)
            classify_result = self.classifier_freq(patch_output_fusion)
        elif self.fusion_type == 'add':
            patch_output_fusion = patch_output_freq + patch_output_time
            classify_result = self.classifier_freq(patch_output_fusion)
        else:
            raise ValueError('fusion_type must be max, mean, gate or add')

        patch_anomaly_score = torch.nn.Sigmoid()(classify_result.repeat(1, seq_length // self.patch_size, 1))
        return classify_result, patch_anomaly_score


class DetectionNetTimeFreqV3(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, n_fft=4):
        super().__init__()
        self.patch_size = patch_size
        self.n_fft = n_fft
        self.proj_size_freq = patch_size * 2 * (self.n_fft // 2 + 1)
        self.proj_size_time = patch_size
        self.hidden_size = expansion_ratio * self.proj_size_freq
        self.proj_freq = nn.Linear(self.proj_size_freq, self.hidden_size)
        self.proj_time = nn.Linear(self.proj_size_time, self.hidden_size)
        self.act = nn.SELU()
        self.encoder_time = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.encoder_freq = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        self.classifier_time = nn.Linear(self.hidden_size * 2, 1)
        self.classifier_freq = nn.Linear(self.hidden_size * 2, 1)
        self.decision_net = nn.Sequential(
            nn.Linear(2, 10),
            nn.SELU(),
            nn.Linear(10, 1)
        )

        nn.init.xavier_uniform_(self.proj_freq.weight)
        nn.init.zeros_(self.proj_freq.bias)
        nn.init.xavier_uniform_(self.proj_time.weight)
        nn.init.zeros_(self.proj_time.bias)
        nn.init.xavier_uniform_(self.classifier_time.weight)
        nn.init.zeros_(self.classifier_time.bias)
        nn.init.xavier_uniform_(self.classifier_freq.weight)
        nn.init.zeros_(self.classifier_freq.bias)

    def forward(self,
                x: torch.Tensor):
        """
        :param x: batch_size, seq_length
        :return:
        """
        # (batch_size * num_channels, seq_len, 2 * (self.n_fft // 2 + 1))
        batch_size, seq_length = x.shape
        x_freq = time_to_timefreq(x, n_fft=self.n_fft, C=1)
        x_freq = x_freq[:, :seq_length, :]
        x_time = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)

        # batch_size * num_channels, patch_num, patch_length * fft_length
        patch_freq = x_freq.view(batch_size, seq_length // self.patch_size, -1)
        patch_time = x_time.view(batch_size, seq_length // self.patch_size, -1)

        # batch_size * num_channels, patch_num, hidden_size
        freq_proj = self.proj_freq(patch_freq)
        freq_proj = self.act(freq_proj)

        time_proj = self.proj_time(patch_time)
        time_proj = self.act(time_proj)

        patch_output_freq, patch_hidden_freq = self.encoder_freq(freq_proj)
        patch_output_time, patch_hidden_time = self.encoder_time(time_proj)

        # patch_output_freq = patch_output_freq.reshape(batch_size * (seq_length // self.patch_size), -1)
        # patch_output_time = patch_output_time.reshape(batch_size * (seq_length // self.patch_size), -1)
        #
        # patch_output = self.fusion_net(patch_output_freq, patch_output_time)

        # batch_size * num_channels, patch_num, 1
        classify_result_freq = self.classifier_freq(patch_output_freq)
        classify_result_time = self.classifier_time(patch_output_time)
        classify_result = torch.cat((classify_result_freq, classify_result_time), dim=-1)
        classify_result = self.decision_net(classify_result)
        patch_anomaly_score = torch.nn.Sigmoid()(classify_result)

        return classify_result, patch_anomaly_score


class DetectionNetConv(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers=8, n_fft=4):
        super().__init__()
        self.patch_size = patch_size
        self.n_fft = n_fft
        self.hidden_time = expansion_ratio * patch_size
        self.hidden_freq = expansion_ratio * patch_size * 2 * (self.n_fft // 2 + 1)

        self.proj_time = nn.Linear(patch_size, expansion_ratio * patch_size)
        self.proj_freq = nn.Conv1d(
            in_channels= 2 * (self.n_fft // 2 + 1) ,
            out_channels=self.hidden_freq,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.encoder_time = nn.GRU(
            input_size=self.hidden_time,
            hidden_size=self.hidden_time,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.encoder_freq = nn.GRU(
            input_size=self.hidden_freq,
            hidden_size=self.hidden_freq,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        self.act = nn.SELU()
        self.classifier_time = nn.Linear(2 * self.hidden_time, 1)
        self.classifier_freq = nn.Linear(2 * self.hidden_freq, 1)

        nn.init.xavier_uniform_(self.proj_time.weight)
        nn.init.zeros_(self.proj_time.bias)
        nn.init.xavier_uniform_(self.classifier_time.weight)
        nn.init.zeros_(self.classifier_time.bias)

        nn.init.xavier_uniform_(self.proj_freq.weight)
        nn.init.zeros_(self.proj_freq.bias)
        nn.init.xavier_uniform_(self.classifier_freq.weight)
        nn.init.zeros_(self.classifier_freq.bias)

    def forward(self,
                x: torch.Tensor):
        """
        :param x: batch_size, seq_length
        :return:
        """
        
        # (batch_size * num_channels, seq_len, 2 * (self.n_fft // 2 + 1))
        batch_size, seq_length = x.shape
        x_freq = time_to_timefreq(x, n_fft=self.n_fft)
        x_freq = x_freq[:, :seq_length, :]
        x_freq = x_freq.permute(0, 2, 1)
        # ( batch_size, 2 * (self.n_fft // 2 + 1), seq_len)
        freq_proj = self.proj_freq(x_freq)
        freq_proj = self.act(freq_proj)
        freq_proj = freq_proj.permute(0, 2, 1)
        patch_output_freq, patch_hidden_freq = self.encoder_freq(freq_proj)

        patch_time = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        time_proj = self.proj_time(patch_time)
        time_proj = self.act(time_proj)
        patch_output_time, patch_hidden_time = self.encoder_time(time_proj)

        classify_result_freq = self.classifier_freq(patch_output_freq)
        classify_result_time = self.classifier_time(patch_output_time)
        classify_result = torch.cat((classify_result_freq, classify_result_time), dim=-1)
        classify_result = torch.max(classify_result, dim=-1, keepdim=True).values
        patch_anomaly_score = torch.nn.Sigmoid()(classify_result)

        return classify_result, patch_anomaly_score

class TransformerReconstructionNet(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, patch_num=None):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = expansion_ratio * patch_size
        self.proj = nn.Linear(patch_size, self.hidden_size)
        self.act = nn.SELU()
        self.encoder = TransformerEncoder(d_model=self.hidden_size, num_layers=num_layers)
        self.head = nn.Linear(self.hidden_size, patch_size)

        self.mask_embedding = nn.Parameter(torch.randn(self.hidden_size), requires_grad=True)
        self.loss_fn = nn.MSELoss(reduction='none')
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,
                x: torch.Tensor,
                patch_anomaly_score: torch.Tensor = None,
                raw_x: torch.Tensor = None,
                patch_labels: torch.Tensor = None,
                is_training=True):
        """
                :param raw_x:
                :param x: batch_size * num_channels, seq_length
                :param patch_anomaly_score: batch_size * num_channels, patch_num, 1
                :param patch_labels: batch_size * num_channels, patch_num, 1
                :return:
                """
        # x: batch_size * num_channels, num_patches, patch_size
        x_patch = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)

        batch_size, num_patches, _ = x_patch.shape

        # x_proj: batch_size * num_channels, num_patches, hidden_size
        patch_proj = self.proj(x_patch)
        patch_proj = self.act(patch_proj)

        # mask_embedding: batch_size * num_channels, num_patches, hidden_size
        mask_embedding = self.mask_embedding.unsqueeze(0).unsqueeze(0)
        mask_embedding = mask_embedding.repeat(batch_size, num_patches, 1)

        # weighted_sum embedding
        # mask_embedding: batch_size * num_channels, num_patches, hidden_size
        patch_anomaly_score = patch_anomaly_score
        soft_mask_patches = mask_embedding * patch_anomaly_score + (1 - patch_anomaly_score) * patch_proj
        patch_out = self.encoder(soft_mask_patches)
        recon_values = self.head(patch_out)

        loss = None
        if is_training:
            raw_x = raw_x.squeeze(-1)
            raw_x_patch = raw_x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
            loss = self.loss_fn(recon_values, raw_x_patch)
            patch_labels = patch_labels.unsqueeze(-1)
            anomaly_loss = loss * patch_labels
            anomaly_loss = torch.sum(anomaly_loss) / (torch.sum(patch_labels) * self.patch_size + 1e-7)
            un_anomaly_loss = loss * (1 - patch_labels)
            un_anomaly_loss = torch.sum(un_anomaly_loss) / (torch.sum(1 - patch_labels) * self.patch_size + 1e-7)
            loss = 0.5 * anomaly_loss + 0.5 * un_anomaly_loss
        # batch_size, seq_length
        recon_values = recon_values.reshape(batch_size, -1)
        return recon_values, loss

class ReconstructionNet(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, patch_num=None):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = expansion_ratio * patch_size
        self.proj = nn.Linear(patch_size, self.hidden_size)
        self.act = nn.SELU()
        if patch_num is None:
            self.encoder = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.1
            )
            self.head = nn.Linear(self.hidden_size * 2, patch_size)
        else:
            self.encoder = PatchMixerEncoder(
                num_layers=num_layers,
                hidden_size=self.hidden_size,
                num_patches=patch_num,
                gated_attn=True,
                expansion_factor=3
            )
            self.head = nn.Linear(self.hidden_size, patch_size)

        self.mask_embedding = nn.Parameter(torch.randn(self.hidden_size), requires_grad=True)
        self.loss_fn = nn.MSELoss(reduction='none')
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,
                x: torch.Tensor,
                patch_anomaly_score: torch.Tensor = None,
                raw_x: torch.Tensor = None,
                patch_labels: torch.Tensor = None,
                is_training=True):
        """
        :param raw_x:
        :param x: batch_size * num_channels, seq_length
        :param patch_anomaly_score: batch_size * num_channels, patch_num, 1
        :param patch_labels: batch_size * num_channels, patch_num, 1
        :return:
        """
        # x: batch_size * num_channels, num_patches, patch_size
        x_patch = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)

        batch_size, num_patches, _ = x_patch.shape

        # x_proj: batch_size * num_channels, num_patches, hidden_size
        patch_proj = self.proj(x_patch)
        patch_proj = self.act(patch_proj)

        # mask_embedding: batch_size * num_channels, num_patches, hidden_size
        mask_embedding = self.mask_embedding.unsqueeze(0).unsqueeze(0)
        mask_embedding = mask_embedding.repeat(batch_size, num_patches, 1)

        # weighted_sum embedding
        # mask_embedding: batch_size * num_channels, num_patches, hidden_size
        patch_anomaly_score = patch_anomaly_score
        # patch_anomaly_score = patch_anomaly_score + 0.1
        # patch_anomaly_score = torch.clamp(patch_anomaly_score, max=1.0)
        soft_mask_patches = mask_embedding * patch_anomaly_score + (1 - patch_anomaly_score) * patch_proj
        patch_out, patch_hidden = self.encoder(soft_mask_patches)
        # patch_out = patch_out + soft_mask_patches
        # batch_size * num_channels, num_patches, patch_size
        recon_values = self.head(patch_out)

        loss = None
        if is_training:
            raw_x = raw_x.squeeze(-1)
            raw_x_patch = raw_x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
            loss = self.loss_fn(recon_values, raw_x_patch)
            patch_labels = patch_labels.unsqueeze(-1)
            anomaly_loss = loss * patch_labels
            anomaly_loss = torch.sum(anomaly_loss) / (torch.sum(patch_labels) * self.patch_size + 1e-7)
            un_anomaly_loss = loss * (1 - patch_labels)
            un_anomaly_loss = torch.sum(un_anomaly_loss) / (torch.sum(1 - patch_labels) * self.patch_size + 1e-7)
            loss = 0.5 * anomaly_loss + 0.5 * un_anomaly_loss
        # batch_size, seq_length
        recon_values = recon_values.reshape(batch_size, -1)
        return recon_values, loss


class ReconstructionNetMixer(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, n_fft=4):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = expansion_ratio * patch_size
        self.proj = nn.Linear(patch_size, self.hidden_size)
        self.act = nn.SELU()
        self.encoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.loss_fn = nn.MSELoss()
        self.head = nn.Linear(self.hidden_size * 2, patch_size)
        self.mask_embedding = nn.Parameter(torch.randn(self.hidden_size), requires_grad=False)
        self.loss_fn = nn.MSELoss(reduction='none')
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,
                x: torch.Tensor,
                patch_anomaly_score: torch.Tensor,
                raw_x: torch.Tensor = None,
                patch_labels: torch.Tensor = None,
                is_training=True):
        """
        :param x: batch_size * num_channels, seq_length
        :param patch_anomaly_score: batch_size * num_channels, patch_num, 1
        :param patch_labels: batch_size * num_channels, patch_num, 1
        :return:
        """
        # x: batch_size * num_channels, num_patches, patch_size
        x_patch = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)

        if is_training:
            raw_x = raw_x.squeeze(-1)
            raw_x_patch = raw_x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)

        batch_size, num_patches, _ = x_patch.shape

        # x_proj: batch_size * num_channels, num_patches, hidden_size
        patch_proj = self.proj(x_patch)
        patch_proj = self.act(patch_proj)

        # mask_embedding: batch_size * num_channels, num_patches, hidden_size
        mask_embedding = self.mask_embedding.unsqueeze(0).unsqueeze(0)
        mask_embedding = mask_embedding.repeat(batch_size, num_patches, 1)

        # weighted_sum embedding
        # mask_embedding: batch_size * num_channels, num_patches, hidden_size
        soft_mask_patches = mask_embedding * patch_anomaly_score + (1 - patch_anomaly_score) * patch_proj
        patch_out, patch_hidden = self.encoder(soft_mask_patches)
        # batch_size * num_channels, num_patches, patch_size
        recon_values = self.head(patch_out)

        loss = None
        if is_training:
            loss = self.loss_fn(recon_values, raw_x_patch)
            loss = loss * patch_labels.unsqueeze(-1)
            if torch.sum(patch_labels) == 0:
                loss = loss.mean()
            else:
                loss = torch.sum(loss) / (torch.sum(patch_labels) * self.patch_size + 1)
            # loss = torch.mean(loss)
        # batch_size, seq_length
        recon_values = recon_values.reshape(batch_size, -1)
        return recon_values, loss


class HardReconstructionNet(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, patch_num=None, mode = None):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = expansion_ratio * patch_size
        self.proj = nn.Linear(patch_size, self.hidden_size)
        self.act = nn.SELU()
        self.mode = mode
        if patch_num is None:
            self.encoder = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.1
            )
            self.head = nn.Linear(self.hidden_size * 2, patch_size)
        else:
            self.encoder = PatchMixerEncoder(
                num_layers=num_layers,
                hidden_size=self.hidden_size,
                num_patches=patch_num,
                gated_attn=True,
                expansion_factor=3
            )
            self.head = nn.Linear(self.hidden_size, patch_size)

        self.mask_embedding = nn.Parameter(torch.randn(self.hidden_size), requires_grad=False)
        self.loss_fn = nn.MSELoss()
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,
                x: torch.Tensor,
                threshold = None,
                patch_anomaly_score: torch.Tensor = None,
                raw_x: torch.Tensor = None,
                patch_labels: torch.Tensor = None,
                is_training= True):

        # x: batch_size * num_channels, num_patches, patch_size
        x_patch = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        device = x.device

        batch_size, num_patches, patch_length = x_patch.shape

        # x_proj: batch_size * num_channels, num_patches, hidden_size
        patch_proj = self.proj(x_patch)
        patch_proj = self.act(patch_proj)

        _, _, hidden_dim = patch_proj.shape

        noise = torch.randn(batch_size, num_patches, device=device)
        mask = torch.ones(batch_size, num_patches, device=device)
        len_keep = int(num_patches * 0.5)
        mask[:, :len_keep] = 0
        ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs * num_channels x L]
        mask = torch.gather(mask, dim=-1, index=ids_restore)  # mask: [bs * num_channels x L]
        mask_0 = mask.unsqueeze(-1).repeat(1, 1, hidden_dim)
        mask_1 = (mask_0 == 0).long()

        mask_proj_0 = patch_proj.masked_fill(mask_0.bool(), 0)
        mask_proj_1 = patch_proj.masked_fill(mask_1.bool(), 0)

        encoder_patch_0 = self.encoder(mask_proj_0)[0]
        encoder_patch_1 = self.encoder(mask_proj_1)[0]

        # batch_size * num_channels, num_patches, patch_size
        recons_0 = self.head(encoder_patch_0)
        recons_1 = self.head(encoder_patch_1)
        mask_0 = mask_0[:, :, :patch_length]
        mask_1 = mask_1[:, :, :patch_length]
        recons = recons_0 * mask_0 + recons_1 * mask_1
        recon_raw = recons_0 * (1 - mask_0) + recons_1 * (1 - mask_1)

        loss = None
        if is_training:
            raw_x = raw_x.squeeze(-1)
            raw_x_patch = raw_x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
            loss = self.loss_fn(recons, raw_x_patch)

            if self.mode == "guide":
                loss_raw = self.loss_fn(recon_raw, raw_x_patch)
                loss = 0.5 * loss + 0.5 * loss_raw

        if threshold is not None and not is_training:
            # patch_anomaly_score: batch_size * num_channels, patch_num, 1
            mask = (patch_anomaly_score > threshold).float()
            masked_patch_proj = patch_proj * (1 -  mask)
            encoder_patch = self.encoder(masked_patch_proj)[0]
            recons = self.head(encoder_patch)

        recons = recons.reshape(batch_size, -1)
        return recons, loss


class GatingReconstructionNet(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers, patch_num=None):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = expansion_ratio * patch_size
        self.proj = nn.Linear(patch_size, self.hidden_size)
        self.act = nn.SELU()
        if patch_num is None:
            self.encoder = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.1
            )
            self.head = nn.Linear(self.hidden_size * 2, patch_size)
        else:
            self.encoder = PatchMixerEncoder(
                num_layers=num_layers,
                hidden_size=self.hidden_size,
                num_patches=patch_num,
                gated_attn=True,
                expansion_factor=3
            )
            self.head = nn.Linear(self.hidden_size, patch_size)

        self.mask_embedding = nn.Parameter(torch.randn(self.hidden_size), requires_grad=False)
        self.loss_fn = nn.MSELoss()
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,
                x: torch.Tensor,
                patch_anomaly_score: torch.Tensor = None,
                raw_x: torch.Tensor = None,
                patch_labels: torch.Tensor = None,
                is_training=True):

        # x: batch_size * num_channels, num_patches, patch_size
        x_patch = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        device = x.device

        batch_size, num_patches, patch_length = x_patch.shape

        # x_proj: batch_size * num_channels, num_patches, hidden_size
        patch_proj = self.proj(x_patch)
        patch_proj = self.act(patch_proj)

        _, _, hidden_dim = patch_proj.shape

        odd_mask = torch.tensor([i % 2 for i in range(num_patches)], device=device).unsqueeze(-1).repeat(1, hidden_dim)
        odd_mask = odd_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        even_mask = 1 - odd_mask

        mask_proj_0 = patch_proj * odd_mask
        mask_proj_1 = patch_proj * even_mask

        encoder_patch_0 = self.encoder(mask_proj_0)[0]
        encoder_patch_1 = self.encoder(mask_proj_1)[0]

        # batch_size * num_channels, num_patches, patch_size
        recons_0 = self.head(encoder_patch_0)
        recons_1 = self.head(encoder_patch_1)
        odd_mask = odd_mask[:, :, :patch_length]
        even_mask = even_mask[:, :, :patch_length]
        recons = recons_0 * even_mask + recons_1 * odd_mask

        loss = None
        if is_training:
            raw_x = raw_x.squeeze(-1)
            raw_x_patch = raw_x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
            loss = self.loss_fn(recons, raw_x_patch)

        recons = recons.reshape(batch_size, -1)
        return recons, loss


class PatchFrequencyMask(nn.Module):
    def __init__(self, patch_size, expansion_ratio, num_layers,
                 n_fft=4, patch_num=None, detect_mode="freq_time_v2", recon_mode="soft", omega=10):
        super().__init__()

        self.patch_size = patch_size
        self.classify_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
        self.recon_loss = nn.MSELoss()
        self.omega = omega
        self.detect_mode = detect_mode
        self.recon_mode = recon_mode

        if detect_mode == "conv":
            self.detection_net = DetectionNetConv(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                n_fft=n_fft
            )
        elif detect_mode == "time":
            self.detection_net = DetectionNetTime(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                n_fft=n_fft
            )
        elif detect_mode == "freq_time":
            self.detection_net = DetectionNetTimeFreq(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                n_fft=n_fft
            )
        elif detect_mode == "freq_time_v2":
            self.detection_net = DetectionNetTimeFreqV2(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                n_fft=n_fft
            )
        elif detect_mode == "freq_time_mean":
            self.detection_net = DetectionNetTimeFreqV2(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                n_fft=n_fft,
                fusion_type = "mean"
            )
        elif detect_mode == "freq_time_gate":
            self.detection_net = DetectionNetTimeFreqV2(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                n_fft=n_fft,
                fusion_type = "gate"
            )
        elif detect_mode == "freq_time_add":
            self.detection_net = DetectionNetTimeFreqV2(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                n_fft=n_fft,
                fusion_type = "add"
            )
        elif detect_mode == "freq_time_v3":
            self.detection_net = DetectionNetTimeFreqV3(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                n_fft=n_fft
            )
        elif detect_mode == "freq_time_point":
            self.detection_net = DetectionNetTimeFreqPoint(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                n_fft=n_fft
            )
        elif detect_mode == "freq_time_window":
            self.detection_net = DetectionNetTimeFreqWindow(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                n_fft=n_fft
            )
        else:
            self.detection_net = DetectionNet(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                n_fft=n_fft
            )

        if recon_mode == "hard":
            self.reconstruction_net = HardReconstructionNet(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                patch_num=patch_num)
        elif recon_mode == "soft":
            self.reconstruction_net = ReconstructionNet(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                patch_num=patch_num
            )
        elif recon_mode == "transformer":
            self.reconstruction_net = TransformerReconstructionNet(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                patch_num=patch_num
            )
        elif recon_mode == "gating":
            self.reconstruction_net = GatingReconstructionNet(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                patch_num=patch_num
            )
        elif recon_mode == "guide_hard":
            self.reconstruction_net = HardReconstructionNet(
                patch_size=self.patch_size,
                expansion_ratio=expansion_ratio,
                num_layers=num_layers,
                patch_num=patch_num,
                mode = "guide"
            )
        else:
            raise ValueError("recon_mode should be 'hard' or 'soft' or 'gating'")

    def forward(self,
                x,
                subsequence_length=None):
        """
        Parameters
        ----------
        x: (batch_size, seq_length, num_channels)

        Returns
        -------

        """
        batch_size, seq_length, num_channels = x.shape
        patch_num = seq_length // self.patch_size
        distorted_functions = [uniform_distort, scale_distort, jitering_distort, mirror_flip]
        # labels (batch_size, length)
        x_distorted, labels = random.choice(distorted_functions)(x, subsequence_length)
        x_distorted = torch.cat([x, x_distorted], dim=0)
        labels = torch.cat([torch.zeros_like(labels), labels], dim=0)
        raw_x = torch.cat([x, x], dim=0)
        # raw_x = x
        batch_size, _, _ = raw_x.shape
        labels = labels.to(x.device)
        x_distorted = x_distorted.permute(0, 2, 1)  # (batch_size, num_channels, seq_len)
        # (batch_size * num_channels, seq_len)
        x_distorted = x_distorted.reshape(batch_size * num_channels, seq_length)
        labels_distorted = labels.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        labels_distorted = torch.sum(labels_distorted, dim=-1)
        # (batch_size, patch_num, 1)
        patch_labels = (labels_distorted > 0).float()
        classify_result, patch_anomaly_score = self.detection_net(x_distorted)
        recon_result, recon_loss = self.reconstruction_net(x=x_distorted, raw_x=raw_x,
                                                           patch_anomaly_score=patch_anomaly_score,
                                                           patch_labels=patch_labels)
        if self.detect_mode == "freq_time_point":
            classify_result = classify_result.reshape(classify_result.shape[0], -1)
            classify_loss = self.classify_loss(input=classify_result,
                                               target=labels)
        elif self.detect_mode == "freq_time_window":
            window_labels = torch.sum(labels, dim=-1, keepdim=True)
            window_labels = (window_labels > 0).float()
            # batch_size * num_channels, 1, 1
            classify_result = classify_result.squeeze(-1)
            classify_loss = self.classify_loss(input=classify_result,
                                               target=window_labels)
        elif self.detect_mode == "freq_time_v2":
            patch_labels = patch_labels.reshape(batch_size * patch_num, -1)
            _, _, classify_residual_result, _ = self.detection_net(x=x_distorted, x_refer=recon_result)
            classify_result = classify_result.reshape(batch_size * patch_num, -1)
            classify_residual_result = classify_residual_result.reshape(batch_size * patch_num, -1)
            classify_loss = self.classify_loss(input=classify_result, target=patch_labels)
            classify_residual_loss = self.classify_loss(input=classify_residual_result, target=patch_labels)
            classify_loss = (classify_loss + classify_residual_loss) / 2
        else:
            classify_result = classify_result.reshape(batch_size * patch_num, -1)
            patch_labels = patch_labels.reshape(batch_size * patch_num, -1)
            classify_loss = self.classify_loss(input=classify_result, target=patch_labels)

        loss = classify_loss + self.omega * recon_loss
        return loss, classify_loss, recon_loss

    def predict(self, x, threshold = None):
        batch_size, seq_length, num_channels = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size * num_channels, seq_length)
        _, classify_anomaly_score = self.detection_net(x)
        recon_values, _ = self.reconstruction_net(x = x, patch_anomaly_score = classify_anomaly_score, is_training=False)
        if self.detect_mode == "freq_time_v2":
            _, _, _, classify_residual_score = self.detection_net(x=x, x_refer=recon_values)
            classify_anomaly_score = (classify_anomaly_score + classify_residual_score) / 2
            recon_values, _ = self.reconstruction_net(x = x, patch_anomaly_score = classify_anomaly_score, is_training=False)
        if self.recon_mode == "guide_hard":
            recon_values, _ = self.reconstruction_net(x = x, patch_anomaly_score = classify_anomaly_score, is_training=False, threshold=threshold)

        recon_values = recon_values.unsqueeze(-1)
        # (batch_size * num_channels, patch_num, patch_size)
        classify_anomaly_score = classify_anomaly_score.repeat(1, 1, self.patch_size)
        # (batch_size, num_channels, seq_length)
        classify_anomaly_score = classify_anomaly_score.reshape(batch_size, num_channels, seq_length)
        # (batch_size, seq_length, num_channels)
        classify_anomaly_score = classify_anomaly_score.permute(0, 2, 1)

        return classify_anomaly_score, recon_values


if __name__ == '__main__':
    sample = torch.rand(64, 1024, 1).to("cuda")
    model = PatchFrequencyMask(
        patch_size=16,
        expansion_ratio=2,
        num_layers=2,
        n_fft=4,
        recon_mode="soft",
        detect_mode="freq_time_v2"
    )
    model = model.to("cuda")
    output = model(sample)
    predict = model.predict(sample, 0.1)
    print(predict[0].shape, predict[1].shape)

