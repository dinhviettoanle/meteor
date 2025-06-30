import torch
import torch.nn.functional as F
from torch import nn

from .transformer_encoder import TransformerEncoderLayerAttn, VAETransformerEncoder
from .transformer_helpers import (
    PositionalEncoding,
    TokenEmbedding,
    generate_causal_mask,
    weights_init,
)


class VAETransformerDecoder(nn.Module):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_ff,
        d_seg_emb,
        dropout=0.1,
        activation="relu",
        cond_mode="in-attn",
        return_attn=False,
    ):
        super(VAETransformerDecoder, self).__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_seg_emb = d_seg_emb
        self.dropout = dropout
        self.activation = activation
        self.cond_mode = cond_mode
        self.return_attn = return_attn

        if cond_mode == "in-attn":
            self.seg_emb_proj = nn.Linear(d_seg_emb, d_model, bias=False)
        elif cond_mode == "pre-attn":
            self.seg_emb_proj = nn.Linear(d_seg_emb + d_model, d_model, bias=False)

        self.decoder_layers = nn.ModuleList()
        TransformerEncoderLayerClass = (
            TransformerEncoderLayerAttn if return_attn else nn.TransformerEncoderLayer
        )

        for i in range(n_layer):
            self.decoder_layers.append(
                TransformerEncoderLayerClass(
                    d_model,
                    n_head,
                    d_ff,
                    dropout,
                    activation,
                    # batch_first=True
                )
            )

            self.attn = None

    def forward(self, x, seg_emb):
        if not hasattr(self, "cond_mode"):
            self.cond_mode = "in-attn"
        attn_mask = generate_causal_mask(x.size(0)).to(x.device)
        # attn_mask = generate_causal_mask(x.size(1)).to(x.device) ### batch_first
        # print (attn_mask.size())

        if self.cond_mode == "in-attn":
            seg_emb = self.seg_emb_proj(seg_emb)
        elif self.cond_mode == "pre-attn":
            x = torch.cat([x, seg_emb], dim=-1)
            x = self.seg_emb_proj(x)

        out = x
        attn_matrices = {}
        for i in range(self.n_layer):
            if self.cond_mode == "in-attn":
                out += seg_emb
            out = self.decoder_layers[i](out, src_mask=attn_mask)

            if self.return_attn:
                attn_matrices[i] = self.decoder_layers[i].attn

        self.attn_matrices = attn_matrices

        return out


class MuseMorphose(nn.Module):
    def __init__(
        self,
        enc_n_layer,
        enc_n_head,
        enc_d_model,
        enc_d_ff,
        dec_n_layer,
        dec_n_head,
        dec_d_model,
        dec_d_ff,
        d_vae_latent,
        d_embed,
        n_token,
        enc_dropout=0.1,
        enc_activation="relu",
        dec_dropout=0.1,
        dec_activation="relu",
        # control
        d_rfreq_emb=32,
        d_polyph_emb=32,
        n_rfreq_cls=8,
        n_polyph_cls=8,
        # control polyph
        d_harmdiv_emb=32,
        d_rhythmdiv_emb=32,
        n_harmdiv_cls=8,
        n_rhythmdiv_cls=8,
        is_training=True,
        use_attr_cls=True,
        use_attr_multitrack_cls=True,
        with_melody_loss=False,
        special_token_idxs=None,
        cond_mode="in-attn",
        return_attn=False,
    ):
        super(MuseMorphose, self).__init__()
        self.enc_n_layer = enc_n_layer
        self.enc_n_head = enc_n_head
        self.enc_d_model = enc_d_model
        self.enc_d_ff = enc_d_ff
        self.enc_dropout = enc_dropout
        self.enc_activation = enc_activation

        self.dec_n_layer = dec_n_layer
        self.dec_n_head = dec_n_head
        self.dec_d_model = dec_d_model
        self.dec_d_ff = dec_d_ff
        self.dec_dropout = dec_dropout
        self.dec_activation = dec_activation

        self.d_vae_latent = d_vae_latent
        self.n_token = n_token
        self.is_training = is_training

        self.cond_mode = cond_mode
        self.token_emb = TokenEmbedding(n_token, d_embed, enc_d_model)
        self.d_embed = d_embed
        self.pe = PositionalEncoding(d_embed)
        self.dec_out_proj = nn.Linear(dec_d_model, n_token)
        self.encoder = VAETransformerEncoder(
            enc_n_layer,
            enc_n_head,
            enc_d_model,
            enc_d_ff,
            d_vae_latent,
            enc_dropout,
            enc_activation,
            return_attn=return_attn,
        )

        self.use_attr_cls = use_attr_cls
        self.use_attr_multitrack_cls = use_attr_multitrack_cls

        d_latent = d_vae_latent
        if use_attr_cls:
            d_latent += d_polyph_emb + d_rfreq_emb
        if use_attr_multitrack_cls:
            d_latent += d_harmdiv_emb  # + d_rhythmdiv_emb # XXX

        self.decoder = VAETransformerDecoder(
            dec_n_layer,
            dec_n_head,
            dec_d_model,
            dec_d_ff,
            d_latent,
            dropout=dec_dropout,
            activation=dec_activation,
            cond_mode=cond_mode,
            return_attn=return_attn,
        )

        if use_attr_cls:
            self.d_rfreq_emb = d_rfreq_emb
            self.d_polyph_emb = d_polyph_emb
            self.rfreq_attr_emb = TokenEmbedding(n_rfreq_cls, d_rfreq_emb, d_rfreq_emb)
            self.polyph_attr_emb = TokenEmbedding(
                n_polyph_cls, d_polyph_emb, d_polyph_emb
            )
        else:
            self.rfreq_attr_emb = None
            self.polyph_attr_emb = None

        if use_attr_multitrack_cls:
            self.d_harmdiv_emb = d_harmdiv_emb
            self.harmdiv_attr_emb = TokenEmbedding(
                n_harmdiv_cls, d_harmdiv_emb, d_harmdiv_emb
            )
        else:
            self.harmdiv_attr_emb = None

        self.with_melody_loss = with_melody_loss
        if with_melody_loss and special_token_idxs is not None:
            self.melody_begin_idx = special_token_idxs["melody_begin_idx"]
            self.melody_end_idx = special_token_idxs["melody_end_idx"]
            self.bar_beat_idxs = special_token_idxs["bar_beat_idxs"]
            self.pad_token = special_token_idxs["pad_token"]

        self.emb_dropout = nn.Dropout(self.enc_dropout)
        self.apply(weights_init)

        if return_attn:
            print("[model] Returning attention matrices")

    def reparameterize(self, mu, logvar, use_sampling=True, sampling_var=1.0):
        std = torch.exp(0.5 * logvar).to(mu.device)
        if use_sampling:
            eps = torch.randn_like(std).to(mu.device) * sampling_var
        else:
            eps = torch.zeros_like(std).to(mu.device)

        return eps * std + mu

    def get_sampled_latent(
        self,
        inp,
        padding_mask=None,
        use_sampling=False,
        sampling_var=0.0,
        return_token_embedding=False,
    ):
        token_emb = self.token_emb(inp)
        enc_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))

        _, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
        mu, logvar = mu.reshape(-1, mu.size(-1)), logvar.reshape(-1, mu.size(-1))
        vae_latent = self.reparameterize(
            mu, logvar, use_sampling=use_sampling, sampling_var=sampling_var
        )

        if return_token_embedding:
            return {"vae_latent": vae_latent, "enc_token_emb": token_emb}
        else:
            return vae_latent

    def generate(
        self,
        inp,
        dec_seg_emb,
        rfreq_cls=None,
        polyph_cls=None,
        harmdiv_cls=None,
        rhythmdiv_cls=None,
        keep_last_only=True,
    ):
        token_emb = self.token_emb(inp)
        dec_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))

        list_concat_embeds = [dec_seg_emb]

        if (
            rfreq_cls is not None
            and polyph_cls is not None
            and self.rfreq_attr_emb is not None
            and self.polyph_attr_emb is not None
        ):
            dec_rfreq_emb = self.rfreq_attr_emb(rfreq_cls)
            dec_polyph_emb = self.polyph_attr_emb(polyph_cls)
            list_concat_embeds.extend([dec_rfreq_emb, dec_polyph_emb])

        if harmdiv_cls is not None and self.harmdiv_attr_emb is not None:
            dec_harmdiv_emb = self.harmdiv_attr_emb(harmdiv_cls)
            list_concat_embeds.extend([dec_harmdiv_emb])

        dec_seg_emb_cat = torch.cat(list_concat_embeds, dim=-1)

        out = self.decoder(dec_inp, dec_seg_emb_cat)
        out = self.dec_out_proj(out)

        if keep_last_only:
            out = out[-1, ...]

        return out

    def forward(
        self,
        enc_inp,
        dec_inp,
        dec_inp_bar_pos,
        rfreq_cls=None,
        polyph_cls=None,
        harmdiv_cls=None,
        rhythmdiv_cls=None,
        padding_mask=None,
        verbose=False,
        return_token_embedding=False,
    ):
        if verbose:
            print("Encoder input", enc_inp.shape)
        if verbose:
            print("Decoder input", dec_inp.shape)
        if verbose:
            print("Decoder input bar positions", dec_inp_bar_pos.shape)

        # [shape of enc_inp] (seqlen_per_bar, bsize, n_bars_per_sample)
        enc_bt_size, enc_n_bars = enc_inp.size(1), enc_inp.size(2)
        # enc_bt_size, enc_n_bars = enc_inp.size(0), enc_inp.size(1) ### batch_first
        enc_token_emb = self.token_emb(enc_inp)
        if verbose:
            print("Encoder embedding inpt", enc_token_emb.shape)

        # [shape of dec_inp] (seqlen_per_sample, bsize)
        # [shape of rfreq_cls & polyph_cls] same as above
        # -- (should copy each bar's label to all corresponding indices)
        dec_token_emb = self.token_emb(dec_inp)
        if verbose:
            print("Decoder embedding inpt", dec_token_emb.shape)

        # enc_token_emb = enc_token_emb.reshape(
        #   enc_inp.size(0), -1, enc_token_emb.size(-1)
        # )
        enc_token_emb = enc_token_emb.reshape(
            enc_inp.size(0), -1, enc_token_emb.size(-1)
        )
        # enc_token_emb = enc_token_emb.reshape(-1, enc_inp.size(2), enc_token_emb.size(-1)) ### batch_first

        enc_inp = self.emb_dropout(enc_token_emb) + self.pe(enc_inp.size(0))
        dec_inp = self.emb_dropout(dec_token_emb) + self.pe(dec_inp.size(0))

        # enc_inp = self.emb_dropout(enc_token_emb) + self.pe(enc_inp.size(2)) ### batch_first
        # dec_inp = self.emb_dropout(dec_token_emb) + self.pe(dec_inp.size(1)) ### batch_first

        if verbose:
            print("Encoder input after PE", enc_inp.shape)
        if verbose:
            print("Decoder input after PE", dec_inp.shape)
        # [shape of padding_mask] (bsize, n_bars_per_sample, seqlen_per_bar)
        # -- should be `True` for padded indices (i.e., those >= seqlen of the bar), `False` otherwise
        if verbose:
            print("Padding mask before reshape", padding_mask.shape)  # type: ignore
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))
            # padding_mask = padding_mask.reshape(enc_bt_size, -1) ### batch_first
        if verbose:
            print("Padding mask after reshape", padding_mask.shape)  # type: ignore

        _, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
        if verbose:
            print("Mu", mu.shape)
        vae_latent = self.reparameterize(mu, logvar)
        if verbose:
            print("VAE latent before reshape", vae_latent.shape)
        vae_latent_reshaped = vae_latent.reshape(enc_bt_size, enc_n_bars, -1)
        if verbose:
            print("VAE latent after reshape", vae_latent_reshaped.shape)

        dec_seg_emb = torch.zeros(
            dec_inp.size(0), dec_inp.size(1), self.d_vae_latent
        ).to(vae_latent.device)
        if verbose:
            print("Decoder segment embedding goal", dec_seg_emb.shape)
        if verbose:
            print("Decoder input bar pos", dec_inp_bar_pos.shape)
        # for n in range(dec_inp.size(0)): ### batch_first
        for n in range(dec_inp.size(1)):
            # [shape of dec_inp_bar_pos] (bsize, n_bars_per_sample + 1)
            # -- stores [[start idx of bar #1, sample #1, ..., start idx of bar #K, sample #1, seqlen of sample #1], [same for another sample], ...]
            for b, (st, ed) in enumerate(
                zip(dec_inp_bar_pos[n, :-1], dec_inp_bar_pos[n, 1:])
            ):
                dec_seg_emb[st:ed, n, :] = vae_latent_reshaped[n, b, :]

        if verbose:
            print("Decoder segment embedding before cat", dec_seg_emb.shape)

        # Decoder input conditions
        list_concat_embeds = [dec_seg_emb]

        if (
            self.use_attr_cls
            and rfreq_cls is not None
            and polyph_cls is not None
            and self.rfreq_attr_emb is not None
            and self.polyph_attr_emb is not None
        ):
            dec_rfreq_emb = self.rfreq_attr_emb(rfreq_cls)
            dec_polyph_emb = self.polyph_attr_emb(polyph_cls)
            list_concat_embeds.extend([dec_rfreq_emb, dec_polyph_emb])

        # if self.use_attr_multitrack_cls and harmdiv_cls is not None and rhythmdiv_cls is not None: # XXX
        #   dec_harmdiv_emb = self.harmdiv_attr_emb(harmdiv_cls)
        #   dec_rhythmdiv_emb = self.rhythmdiv_attr_emb(rhythmdiv_cls)
        #   list_concat_embeds.extend([dec_harmdiv_emb, dec_rhythmdiv_emb])
        if (
            self.use_attr_multitrack_cls
            and harmdiv_cls is not None
            and self.harmdiv_attr_emb is not None
        ):
            dec_harmdiv_emb = self.harmdiv_attr_emb(harmdiv_cls)
            list_concat_embeds.extend([dec_harmdiv_emb])

        dec_seg_emb_cat = torch.cat(list_concat_embeds, dim=-1)

        if verbose:
            print("!! Decoder segment embedding after cat", dec_seg_emb_cat.shape)
        dec_out = self.decoder(dec_inp, dec_seg_emb_cat)
        dec_logits = self.dec_out_proj(dec_out)

        if verbose:
            print("Decoder out", dec_out.shape)
        if verbose:
            print("Decoder logits", dec_logits.shape)

        if verbose:
            print("============================================")

        if return_token_embedding:
            return {
                "out": (mu, logvar, dec_logits),
                "enc_token_emb": enc_token_emb,
                "dec_token_emb": dec_token_emb,
            }
        else:
            return mu, logvar, dec_logits

    def compute_loss(
        self,
        mu,
        logvar,
        beta,
        fb_lambda,
        dec_logits,
        dec_tgt,
        batch_output_melody_logits=None,
        batch_melody_tokens=None,
        melody_weight=None,
    ):
        recons_loss = F.cross_entropy(
            dec_logits.view(-1, dec_logits.size(-1)),
            dec_tgt.contiguous().view(-1),
            ignore_index=self.n_token - 1,
            reduction="mean",
        ).float()

        kl_raw = -0.5 * (1 + logvar - mu**2 - logvar.exp()).mean(dim=0)
        kl_before_free_bits = kl_raw.mean()
        kl_after_free_bits = kl_raw.clamp(min=fb_lambda)
        kldiv_loss = kl_after_free_bits.mean()

        if (
            melody_weight is not None
            and batch_melody_tokens is not None
            and batch_output_melody_logits is not None
        ):
            melody_loss = F.cross_entropy(
                batch_output_melody_logits.view(-1, dec_logits.shape[-1]),
                batch_melody_tokens.contiguous().view(-1),
                ignore_index=self.n_token - 1,
                reduction="mean",
            ).float()
        else:
            melody_loss = torch.tensor(0)
            melody_weight = 0

        return {
            "beta": beta,
            "total_loss": recons_loss + beta * kldiv_loss + melody_weight * melody_loss,
            "kldiv_loss": kldiv_loss,
            "kldiv_raw": kl_before_free_bits,
            "recons_loss": recons_loss,
            "melody_loss": melody_loss,
        }
