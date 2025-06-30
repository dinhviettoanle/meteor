from torch import nn
from torch import Tensor
from typing import Optional


class TransformerEncoderLayerAttn(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs) -> None:
        super(TransformerEncoderLayerAttn, self).__init__(*args, **kwargs)
        self.attn = None

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x, attn = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        self.attn = attn
        return self.dropout1(x)


# ============================================================================================


class VAETransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_ff,
        d_vae_latent,
        dropout=0.1,
        activation="relu",
        return_attn=False,
    ):
        super(VAETransformerEncoder, self).__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_vae_latent = d_vae_latent
        self.dropout = dropout
        self.activation = activation

        self.return_attn = return_attn
        TransformerEncoderLayerClass = (
            TransformerEncoderLayerAttn if return_attn else nn.TransformerEncoderLayer
        )

        self.tr_encoder_layer = TransformerEncoderLayerClass(
            d_model,
            n_head,
            d_ff,
            dropout,
            activation,
            # batch_first=True
        )
        self.tr_encoder = nn.TransformerEncoder(self.tr_encoder_layer, n_layer)

        self.fc_mu = nn.Linear(d_model, d_vae_latent)
        self.fc_logvar = nn.Linear(d_model, d_vae_latent)

        if return_attn:
            self.attn_matrices = None

    def forward(self, x, padding_mask=None):
        out = self.tr_encoder(x, src_key_padding_mask=padding_mask)
        hidden_out = out[0, :, :]
        # hidden_out = out[:, 0, :] ### batch_first
        mu, logvar = self.fc_mu(hidden_out), self.fc_logvar(hidden_out)

        if self.return_attn:
            attn_matrices = {}
            for ii, layer in enumerate(self.tr_encoder.layers):
                attn_matrices[ii] = layer.attn
            self.attn_matrices = attn_matrices

        return hidden_out, mu, logvar
