import os
import shutil
import sys
import time
from pprint import pprint

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataloader import FullSongToMelodyDataset, REMIFullSongTransformerDataset
from .model.musemorphose import MuseMorphose
from .utils import pickle_load

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    pprint(config)
else:
    config = None

if config is not None:
    device = config["training"]["device"]
    device = torch.device(device)
    print("Using", device)
    trained_steps = config["training"]["trained_steps"]
    lr_decay_steps = config["training"]["lr_decay_steps"]
    lr_warmup_steps = config["training"]["lr_warmup_steps"]
    no_kl_steps = config["training"]["no_kl_steps"]
    kl_cycle_steps = config["training"]["kl_cycle_steps"]
    no_melody_steps = config["training"].get("no_melody_steps", 0)
    kl_max_beta = config["training"]["kl_max_beta"]
    free_bit_lambda = config["training"]["free_bit_lambda"]
    max_lr, min_lr = config["training"]["max_lr"], config["training"]["min_lr"]

    ckpt_dir = config["training"]["ckpt_dir"]
    params_dir = os.path.join(ckpt_dir, "params/")
    optim_dir = os.path.join(ckpt_dir, "optim/")
    pretrained_params_path = config["model"]["pretrained_params_path"]
    pretrained_optim_path = config["model"]["pretrained_optim_path"]
    ckpt_interval = config["training"]["ckpt_interval"]
    log_interval = config["training"]["log_interval"]
    val_interval = config["training"]["val_interval"]
    constant_kl = config["training"]["constant_kl"]

    recons_loss_ema = 0.0
    kl_loss_ema = 0.0
    kl_raw_ema = 0.0
    melody_loss_ema = 0.0


# =======================================================================================================
# =======================================================================================================


def log_epoch(log_file, log_data, is_init=False):
    """
    Pretty-print the log for an epoch at training
    """
    if is_init:
        with open(log_file, "w") as f:
            f.write(
                "{:4} {:8} {:12} {:12} {:12} {:12} {:12} {:12} {:12}\n".format(
                    "ep",
                    "steps",
                    "lr",
                    "recons_loss",
                    "kldiv_loss",
                    "kl_beta",
                    "kldiv_raw",
                    "melody",
                    "ep_time",
                )
            )

    with open(log_file, "a") as f:
        f.write(
            "{:<4} {:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(
                log_data["ep"],
                log_data["steps"],
                round(log_data["lr"], 6),
                round(log_data["recons_loss"], 5),
                round(log_data["kldiv_loss"], 5),
                round(log_data["kl_beta"], 3),
                round(log_data["kldiv_raw"], 5),
                round(log_data["melody_loss"], 5),
                round(log_data["time"], 2),
            )
        )


def log_valloss(
    epoch,
    trained_steps,
    recons_loss_ema,
    kl_raw_ema,
    melody_loss_ema,
    vallosses,
    is_init=False,
):
    """
    Pretty-print the log for an epoch at validation
    """
    if is_init:
        with open(os.path.join(ckpt_dir, "valloss.txt"), "a") as f:
            f.write(
                "{:4} {:8} {:.12} {:12} {:12} {:12} {:12} {:12}\n".format(
                    "ep",
                    "steps",
                    "tr_rec",
                    "tr_kl",
                    "tr_mel",
                    "val_rec",
                    "val_kl",
                    "val_mel",
                )
            )

    with open(os.path.join(ckpt_dir, "valloss.txt"), "a") as f:
        f.write(
            "{:<4} {:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(
                epoch,
                trained_steps,
                round(recons_loss_ema, 5),
                round(kl_raw_ema, 5),
                round(melody_loss_ema, 5),
                round(np.mean(vallosses[0]), 5),
                round(np.mean(vallosses[1]), 5),
                round(np.mean(vallosses[2]), 5),
            )
        )


# ===============================================================================================


def beta_cyclical_sched(step):
    step_in_cycle = (step - 1) % kl_cycle_steps
    cycle_progress = step_in_cycle / kl_cycle_steps

    if step < no_kl_steps:
        return 0.0
    if cycle_progress < 0.5:
        return kl_max_beta * cycle_progress * 2.0
    else:
        return kl_max_beta


def compute_loss_ema(ema, batch_loss, decay=0.95):
    if ema == 0.0:
        return batch_loss
    else:
        return batch_loss * (1 - decay) + ema * decay


# =================================================================================================


def get_attr_classes(batch_samples, use_attr_cls, use_attr_multitrack_cls, device):
    if use_attr_cls:
        batch_rfreq_cls = batch_samples["rhymfreq_cls"].permute(1, 0).to(device)
        batch_polyph_cls = batch_samples["polyph_cls"].permute(1, 0).to(device)
    else:
        batch_rfreq_cls = None
        batch_polyph_cls = None

    if use_attr_multitrack_cls:
        batch_harmdiv_cls = batch_samples["harmdiv_cls"].permute(1, 0).to(device)
        # batch_rhythmdiv_cls = batch_samples['rhythmdiv_cls'].permute(1, 0).to(device) # XXX
        batch_rhythmdiv_cls = None  # XXX
    else:
        batch_harmdiv_cls = None
        batch_rhythmdiv_cls = None

    return batch_rfreq_cls, batch_polyph_cls, batch_harmdiv_cls, batch_rhythmdiv_cls


# =======================================================================================================


def make_melody_mask_batch(tokens, melody_begin_idx, melody_end_idx, bar_beat_idxs):
    # tokens shape: (sequence_length, batch_size)
    seq_len, batch_size = tokens.shape
    melodic_mask = torch.zeros((seq_len, batch_size), dtype=int)  # type: ignore

    mask = torch.isin(tokens, bar_beat_idxs)
    melodic_mask[mask] = 1

    for b in range(batch_size):
        melody_starts = (tokens[:, b] == melody_begin_idx).nonzero(as_tuple=True)[0]
        melody_ends = (tokens[:, b] == melody_end_idx).nonzero(as_tuple=True)[0]

        for start, end in zip(melody_starts, melody_ends):
            melodic_mask[start : end + 1, b] = 1

    return melodic_mask


def compute_batch_melody_logits(
    dec_logits,
    melody_begin_idx,
    melody_end_idx,
    bar_beat_idxs,
    pad_token,
    padded_seq_len=2500,
):
    seq_len, batch_size, hidden_dim = dec_logits.shape
    tokens = dec_logits.argmax(axis=2)
    melody_mask = make_melody_mask_batch(
        tokens, melody_begin_idx, melody_end_idx, bar_beat_idxs
    )

    batch_output_melody_tokens = pad_token * torch.ones((padded_seq_len, batch_size))
    batch_output_melody_logits = pad_token * torch.ones(
        (padded_seq_len, batch_size, hidden_dim)
    )
    for b in range(batch_size):
        minibatch_melody = tokens[:, b][melody_mask[:, b].bool()]
        batch_output_melody_tokens[: minibatch_melody.shape[0], b] = minibatch_melody
        minibatch_logits = dec_logits[:, b, :][melody_mask[:, b].bool()]
        batch_output_melody_logits[: minibatch_logits.shape[0], b, :] = minibatch_logits

    return batch_output_melody_logits


# =======================================================================================================


def train_model(
    epoch,
    model,
    dloader,
    dloader_val,
    optim,
    sched,
    use_attr_cls=True,
    use_attr_multitrack_cls=True,
    with_melody_tokens=False,
):
    model.train()

    print("[epoch {:03d}] training ...".format(epoch))
    print("[epoch {:03d}] # batches = {}".format(epoch, len(dloader)))
    st = time.time()

    pbar = tqdm(total=len(dloader))
    for batch_idx, batch_samples in enumerate(dloader):
        model.zero_grad()
        batch_enc_inp = batch_samples["enc_input"].permute(2, 0, 1).to(device)
        batch_dec_inp = batch_samples["dec_input"].permute(1, 0).to(device)
        batch_dec_tgt = batch_samples["dec_target"].permute(1, 0).to(device)
        batch_inp_bar_pos = batch_samples["bar_pos"].to(device)
        batch_inp_lens = batch_samples["length"]
        batch_padding_mask = batch_samples["enc_padding_mask"].to(device)
        batch_rfreq_cls, batch_polyph_cls, batch_harmdiv_cls, batch_rhythmdiv_cls = (
            get_attr_classes(
                batch_samples, use_attr_cls, use_attr_multitrack_cls, device
            )
        )
        if with_melody_tokens:
            batch_melody_tokens = (
                batch_samples["melody_tokens"].permute(1, 0).to(device)
            )

        global trained_steps
        trained_steps += 1

        mu, logvar, dec_logits = model(
            batch_enc_inp,
            batch_dec_inp,
            batch_inp_bar_pos,
            batch_rfreq_cls,
            batch_polyph_cls,
            batch_harmdiv_cls,
            batch_rhythmdiv_cls,
            padding_mask=batch_padding_mask,
            verbose=False,  ## DEBUG
        )

        ## beta term
        if not constant_kl:
            kl_beta = beta_cyclical_sched(trained_steps)
        else:
            kl_beta = kl_max_beta

        if with_melody_tokens:
            batch_output_melody_logits = compute_batch_melody_logits(
                dec_logits,
                model.melody_begin_idx,
                model.melody_end_idx,
                model.bar_beat_idxs,
                model.pad_token,
            )
            batch_output_melody_logits = batch_output_melody_logits.to(device)
            if trained_steps > no_melody_steps:
                melody_weight = 1
            else:
                melody_weight = 0
        else:
            batch_output_melody_logits = None
            batch_melody_tokens = None
            melody_weight = None

        if isinstance(model, nn.DataParallel):
            losses = model.module.compute_loss(
                mu,
                logvar,
                kl_beta,
                free_bit_lambda,
                dec_logits,
                batch_dec_tgt,
                batch_output_melody_logits=batch_output_melody_logits,
                batch_melody_tokens=batch_melody_tokens,
                melody_weight=melody_weight,
            )
        else:
            losses = model.compute_loss(
                mu,
                logvar,
                kl_beta,
                free_bit_lambda,
                dec_logits,
                batch_dec_tgt,
                batch_output_melody_logits=batch_output_melody_logits,
                batch_melody_tokens=batch_melody_tokens,
                melody_weight=melody_weight,
            )

        # anneal learning rate
        if trained_steps < lr_warmup_steps:
            curr_lr = max_lr * trained_steps / lr_warmup_steps
            optim.param_groups[0]["lr"] = curr_lr
        else:
            # sched.step(trained_steps - lr_warmup_steps) # deprecated?
            sched.step()

        # clip gradient & update model
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()

        global recons_loss_ema, kl_loss_ema, kl_raw_ema, melody_loss_ema
        recons_loss_ema = compute_loss_ema(
            recons_loss_ema, losses["recons_loss"].item()
        )
        kl_loss_ema = compute_loss_ema(kl_loss_ema, losses["kldiv_loss"].item())
        kl_raw_ema = compute_loss_ema(kl_raw_ema, losses["kldiv_raw"].item())
        melody_loss_ema = compute_loss_ema(
            melody_loss_ema, losses["melody_loss"].item()
        )

        pbar.set_description("epoch {:03d} | batch {:03d}".format(epoch, batch_idx))

        ### ----- Logging -----
        if not trained_steps % log_interval:
            log_data = {
                "ep": epoch,
                "steps": trained_steps,
                "lr": optim.param_groups[0]["lr"],
                "recons_loss": recons_loss_ema,
                "kldiv_loss": kl_loss_ema,
                "kl_beta": kl_beta,
                "kldiv_raw": kl_raw_ema,
                "melody_loss": melody_loss_ema,
                "time": time.time() - st,
            }
            log_epoch(
                os.path.join(ckpt_dir, "log.txt"),
                log_data,
                is_init=not os.path.exists(os.path.join(ckpt_dir, "log.txt")),
            )

        ### ===== Validation =====
        if not trained_steps % val_interval:
            vallosses = validate(
                model,
                dloader_val,
                use_attr_cls=use_attr_cls,
                use_attr_multitrack_cls=use_attr_multitrack_cls,
                with_melody_tokens=with_melody_tokens,
            )
            print(
                "\n\t * [validation] RC: {:.4f} | KL: {:.4f}".format(
                    np.mean(vallosses[0]), np.mean(vallosses[1])
                )
            )
            log_valloss(
                epoch,
                trained_steps,
                recons_loss_ema,
                kl_raw_ema,
                melody_loss_ema,
                vallosses,
                is_init=not os.path.exists(os.path.join(ckpt_dir, "valloss.txt")),
            )
            model.train()

        ### ----- Checkpoint save  -----
        if not trained_steps % ckpt_interval:
            state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            torch.save(
                state_dict,
                os.path.join(
                    params_dir,
                    "step_{:d}-model.pt".format(
                        trained_steps,
                    ),
                ),
            )
            torch.save(
                {
                    "optimizer": optim.state_dict(),
                    "scheduler": sched.state_dict(),
                },
                os.path.join(
                    optim_dir,
                    "step_{:d}-optim.pt".format(
                        trained_steps,
                    ),
                ),
            )

        pbar.update(1)
    pbar.close()

    print("Saved to", ckpt_dir)
    print(
        "[epoch {:03d}] training completed\n  -- loss = (RC: {:.4f} | KL: {:.4f} | KL_raw: {:.4f})\n  -- time elapsed = {:.2f} secs.".format(
            epoch, recons_loss_ema, kl_loss_ema, kl_raw_ema, time.time() - st
        )
    )


# =================================================================================================


def validate(
    model,
    dloader,
    n_rounds=1,
    use_attr_cls=True,
    use_attr_multitrack_cls=True,
    with_melody_tokens=False,
):
    model.eval()
    loss_rec = []
    kl_loss_rec = []
    mel_loss_rec = []

    # print ('[info] validating ...')
    with torch.no_grad():
        pbar_val = tqdm(
            total=n_rounds * len(dloader),
            leave=False,
            desc="Validation ({} rounds x {} pieces)".format(n_rounds, len(dloader)),
        )
        for i in range(n_rounds):
            # print ('[round {}]'.format(i+1))

            for batch_idx, batch_samples in enumerate(dloader):
                model.zero_grad()

                batch_enc_inp = batch_samples["enc_input"].permute(2, 0, 1).to(device)
                batch_dec_inp = batch_samples["dec_input"].permute(1, 0).to(device)
                batch_dec_tgt = batch_samples["dec_target"].permute(1, 0).to(device)
                batch_inp_bar_pos = batch_samples["bar_pos"].to(device)
                batch_padding_mask = batch_samples["enc_padding_mask"].to(device)
                (
                    batch_rfreq_cls,
                    batch_polyph_cls,
                    batch_harmdiv_cls,
                    batch_rhythmdiv_cls,
                ) = get_attr_classes(
                    batch_samples, use_attr_cls, use_attr_multitrack_cls, device
                )
                if with_melody_tokens:
                    batch_melody_mask = (
                        batch_samples["melody_mask"].permute(1, 0).to(device)
                    )
                    batch_melody_tokens = (
                        batch_samples["melody_tokens"].permute(1, 0).to(device)
                    )

                mu, logvar, dec_logits = model(
                    batch_enc_inp,
                    batch_dec_inp,
                    batch_inp_bar_pos,
                    batch_rfreq_cls,
                    batch_polyph_cls,
                    batch_harmdiv_cls,
                    batch_rhythmdiv_cls,
                    padding_mask=batch_padding_mask,
                )

                if with_melody_tokens:
                    batch_output_melody_logits = compute_batch_melody_logits(
                        dec_logits,
                        model.melody_begin_idx,
                        model.melody_end_idx,
                        model.bar_beat_idxs,
                        model.pad_token,
                    )
                    batch_output_melody_logits = batch_output_melody_logits.to(device)
                    if trained_steps > no_melody_steps:
                        melody_weight = 1
                    else:
                        melody_weight = 0
                else:
                    batch_output_melody_logits = None
                    batch_melody_tokens = None
                    melody_weight = None

                if isinstance(model, nn.DataParallel):
                    losses = model.module.compute_loss(
                        mu,
                        logvar,
                        0.0,
                        0.0,
                        dec_logits,
                        batch_dec_tgt,
                        batch_output_melody_logits=batch_output_melody_logits,
                        batch_melody_tokens=batch_melody_tokens,
                        melody_weight=melody_weight,
                    )
                else:
                    losses = model.compute_loss(
                        mu,
                        logvar,
                        0.0,
                        0.0,
                        dec_logits,
                        batch_dec_tgt,
                        batch_output_melody_logits=batch_output_melody_logits,
                        batch_melody_tokens=batch_melody_tokens,
                        melody_weight=melody_weight,
                    )

                loss_rec.append(losses["recons_loss"].item())
                kl_loss_rec.append(losses["kldiv_raw"].item())
                mel_loss_rec.append(losses["melody_loss"].item())
                pbar_val.update(1)
    pbar_val.close()
    return loss_rec, kl_loss_rec, mel_loss_rec


# =================================================================================================
# =================================================================================================
# =================================================================================================

if __name__ == "__main__":
    if config is None:
        raise FileNotFoundError("Config file has not been set")

    mconf = config["model"]

    ### ===== DATASET =====
    with_melody_tokens = config["data"].get("melody_tokens", False)
    ClsDataset = (
        FullSongToMelodyDataset
        if with_melody_tokens
        else REMIFullSongTransformerDataset
    )

    dset = ClsDataset(
        config["data"]["data_dir"],
        config["data"]["vocab_path"],
        do_augment=config["data"].get("do_augment", True),
        model_enc_seqlen=config["data"]["enc_seqlen"],
        model_dec_seqlen=config["data"]["dec_seqlen"],
        model_max_bars=config["data"]["max_bars"],
        pieces=pickle_load(config["data"]["train_split"]),
        pad_to_same=True,
        use_attr_cls=mconf["use_attr_cls"],
        use_attr_multitrack_cls=mconf["use_attr_multitrack_cls"],
        min_pitch=1,
        max_pitch=127,
        files_limit=config["data"].get("files_limit", None),
        melody_first=config["data"].get("melody_first", False),
        melody_mask_dir=config["data"].get("melody_mask_dir", None),
    )
    dset_val = ClsDataset(
        config["data"]["data_dir"],
        config["data"]["vocab_path"],
        do_augment=False,
        model_enc_seqlen=config["data"]["enc_seqlen"],
        model_dec_seqlen=config["data"]["dec_seqlen"],
        model_max_bars=config["data"]["max_bars"],
        pieces=pickle_load(config["data"]["val_split"]),
        pad_to_same=True,
        use_attr_cls=mconf["use_attr_cls"],
        use_attr_multitrack_cls=mconf["use_attr_multitrack_cls"],
        min_pitch=1,
        max_pitch=127,
        files_limit=config["data"].get("files_limit", None),
        melody_first=config["data"].get("melody_first", False),
        melody_mask_dir=config["data"].get("melody_mask_dir", None),
    )
    print("[info]", "# training samples:", len(dset.pieces))

    dloader = DataLoader(
        dset, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=8
    )
    dloader_val = DataLoader(
        dset_val, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=8
    )

    if with_melody_tokens and isinstance(dset, FullSongToMelodyDataset):
        special_token_idxs = {
            "melody_begin_idx": dset.melody_begin_idx,
            "melody_end_idx": dset.melody_end_idx,
            "bar_beat_idxs": dset.bar_beat_idxs,
            "pad_token": dset.pad_token,
        }
    else:
        special_token_idxs = None

    ### ===== MODEL INITIALIZATION =====
    model = MuseMorphose(
        mconf["enc_n_layer"],
        mconf["enc_n_head"],
        mconf["enc_d_model"],
        mconf["enc_d_ff"],
        mconf["dec_n_layer"],
        mconf["dec_n_head"],
        mconf["dec_d_model"],
        mconf["dec_d_ff"],
        mconf["d_latent"],
        mconf["d_embed"],
        dset.vocab_size,
        d_polyph_emb=mconf["d_polyph_emb"],
        d_rfreq_emb=mconf["d_rfreq_emb"],
        d_harmdiv_emb=mconf["d_harmdiv_emb"],
        d_rhythmdiv_emb=mconf["d_rhythmdiv_emb"],
        cond_mode=mconf["cond_mode"],
        use_attr_cls=mconf["use_attr_cls"],
        use_attr_multitrack_cls=mconf["use_attr_multitrack_cls"],
        with_melody_loss=with_melody_tokens,
        special_token_idxs=special_token_idxs,
    ).to(device)

    if pretrained_params_path:
        model.load_state_dict(torch.load(pretrained_params_path, map_location="cpu"))
        model = model.to(device)

    if config["training"].get("parallel_devices"):
        print("!! Using parallel devices !!")
        model = nn.DataParallel(
            model, device_ids=config["training"]["parallel_devices"], dim=1
        )

    if with_melody_tokens:
        model.bar_beat_idxs = model.bar_beat_idxs.to(
            device
        )  # for melody loss computing

    print(model)

    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[info] model # params: {:,}".format(n_params))

    ### ===== OPTIMIZERS INITIALIZATION =====
    opt_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(opt_params, lr=max_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, lr_decay_steps, eta_min=min_lr
    )
    if pretrained_optim_path:
        optim_schd_state_dicts = torch.load(pretrained_optim_path, map_location="cpu")
        optimizer.load_state_dict(optim_schd_state_dicts["optimizer"])
        scheduler.load_state_dict(optim_schd_state_dicts["scheduler"])

    ### ===== SAVE PATH CHECK =====
    if (
        os.path.exists(ckpt_dir)
        and not (pretrained_optim_path)
        and not (pretrained_params_path)
    ):
        if not (len(sys.argv) == 3 and sys.argv[-1] == "-y"):
            input(f"Are you sure you to remove the folder {ckpt_dir}?")
        print("!!!!! REMOVING {}".format(ckpt_dir))
        shutil.rmtree(ckpt_dir)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    if not os.path.exists(optim_dir):
        os.makedirs(optim_dir)
    shutil.copy(config_path, ckpt_dir)
    shutil.copy(config["data"]["vocab_path"], ckpt_dir)

    start_epoch = config["training"].get("start_epoch", 0)

    ### ===== TRAINING LOOP =====
    print("================ START TRAINING ===============")
    for ep in range(start_epoch, config["training"]["max_epochs"]):
        train_model(
            ep + 1,
            model,
            dloader,
            dloader_val,
            optimizer,
            scheduler,
            use_attr_cls=mconf["use_attr_cls"],
            use_attr_multitrack_cls=mconf["use_attr_multitrack_cls"],
            with_melody_tokens=with_melody_tokens,
        )
