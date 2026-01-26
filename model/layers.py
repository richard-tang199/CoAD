import torch
import torch.nn as nn
# from flash_attn.modules.mha import MHA

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]

# class FlashTransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dropout=0.1, dim_feedforward=2048):
#         super().__init__()
#         self.self_attn = MHA(embed_dim=d_model, num_heads=nhead, dropout=dropout)
#
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = nn.ReLU()
#
#     def forward(self, x, key_padding_mask=None):
#         x_res = x
#         x = self.self_attn(x, key_padding_mask=key_padding_mask)
#         x = self.dropout1(x)
#         x = self.norm1(x + x_res)
#
#         x_res = x
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         x = self.dropout2(x)
#         x = self.norm2(x + x_res)
#
#         return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        self.pos_encoder = LearnablePositionalEncoding(max_len=5000, d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = (
            nn.TransformerEncoder(encoder_layer, num_layers=num_layers))

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=None)
        return output




class Patchify(nn.Module):
    def __init__(self, sequence_length, patch_length, patch_stride):
        super().__init__()

        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride

        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

        # get the number of patches
        self.num_patches = (max(self.sequence_length, self.patch_length) - self.patch_length) // self.patch_stride + 1
        new_sequence_length = self.patch_length + self.patch_stride * (self.num_patches - 1)
        self.sequence_start = self.sequence_length - new_sequence_length

    def forward(self, time_values: torch.Tensor):
        """
        Parameters:
            time_values (`torch.Tensor` of shape `(batch_size, num_channels, num_subsequences, sequence_length)`, *required*):
                Input for Patchify

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_subsequences, num_patches, patch_length)`
        """
        sequence_length = time_values.shape[-1]
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
            )
        # output: [bs x num_channels x num_subsequences x new_sequence_length]
        output = time_values[..., self.sequence_start:]
        # output: [bs x num_channels x num_subsequences x num_patches x patch_length]
        output = output.unfold(dimension=-1, size=self.patch_length, step=self.patch_stride)
        return output


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (batch, num_channels, num_subsequences, hidden_dim)
            2. flatten input to (batch*num_channles*num_subsequences, hidden_dim)

        """
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # calculate the distance between embeddings
        e_d = torch.norm(self.embedding.weight[:, None] - self.embedding.weight, dim=2, p=2)
        e_loss = torch.mean(torch.tril(e_d))
        e_loss = torch.exp(-e_loss / 0.1)

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
               torch.mean((z_q - z.detach()) ** 2)
        loss += e_loss

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return loss, z_q, perplexity, min_encoding_indices.view(z.shape[:-1])

# class TransformerFlashEncoder(nn.Module):
#     def __init__(self, d_model, num_layers=6):
#         super().__init__()
#         self.pos_encoder = LearnablePositionalEncoding(max_len=5000, d_model=d_model)
#
#         self.layers = nn.ModuleList([
#             FlashTransformerEncoderLayer(
#                 d_model=d_model,
#                 nhead=4,
#                 dropout=0.1,
#                 dim_feedforward=d_model
#             )
#             for _ in range(num_layers)
#         ])
#
#     def forward(self, x, key_padding_mask=None):
#         x = self.pos_encoder(x)
#         for layer in self.layers:
#             x = layer(x, key_padding_mask=key_padding_mask)
#         return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_layers=4):
        super().__init__()
        self.pos_encoder = LearnablePositionalEncoding(max_len=5000, d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=3,
            dim_feedforward=d_model,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=None)
        return output

if __name__ == '__main__':
    patchify = Patchify(sequence_length=128, patch_length=8, patch_stride=8)
    sample_input = torch.rand(2, 1, 4, 128)
    output = patchify(sample_input)
    print(output.shape)
