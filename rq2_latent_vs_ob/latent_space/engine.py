from __future__ import annotations

import torch
import torch.nn as nn


def assemble_batch(batch, embedder, target_ae, n_latent, label_len, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    th, tf = batch["target_history"], batch["target_future"]
    L, H = th.shape[1], tf.shape[1]

    z_hist = target_ae.encode(th.float())

    cov_hist = embedder.build_covariate(
        batch["continuous_history"], batch["categorical_history"], batch["exog_history"])
    cov_fut = embedder.build_covariate(
        batch["continuous_future"], batch["categorical_future"], batch["exog_future"])

    model_input = torch.cat([z_hist, cov_hist], dim=-1)

    B = z_hist.shape[0]
    series_dec = z_hist.new_zeros((B, label_len + H, n_latent))
    series_dec[:, :label_len] = z_hist[:, L - label_len:]
    cov_full = torch.cat([cov_hist, cov_fut], dim=1)
    cov_dec = cov_full[:, L - label_len:]
    target = torch.cat([series_dec, cov_dec], dim=-1)

    mark_full = torch.cat([batch["mark_history"], batch["mark_future"]], dim=1)
    mark_enc = batch["mark_history"]
    mark_dec = mark_full[:, L - label_len:]
    return model_input, target, mark_enc, mark_dec, cov_fut, tf


def forward_loss(adapter, embedder, target_ae, batch, n_latent, n_target, device, criterion,
                 jepa=None, jepa_alpha: float = 0.0):
    label_len = adapter.config.label_len
    model_input, target, mark_enc, mark_dec, cov_future, target_future = assemble_batch(
        batch, embedder, target_ae, n_latent, label_len, device)

    out = adapter._process(model_input, target, mark_enc, mark_dec, cov_future)
    z_pred = out["output"][:, -target_future.shape[1]:, :n_latent]
    if adapter.CovariateFusion is not None:
        z_pred = adapter.CovariateFusion(cov_future, z_pred)

    pred_obs = target_ae.decode(z_pred)
    target_future = target_future.float()

    main = criterion(pred_obs, target_future)
    loss = main
    aux_val = 0.0
    extra = out.get("additional_loss")
    if extra is not None and torch.is_tensor(extra):
        aux = extra.mean()
        loss = main + aux
        aux_val = aux.item()

    jepa_val = 0.0
    if jepa is not None:
        from latent_space.jepa import jepa_loss

        jl = jepa_loss(z_pred, target_future, jepa)
        jepa_val = jl.item()
        if jepa_alpha:
            loss = loss + jepa_alpha * jl
    stats = {"main": main.item(), "aux": aux_val, "jepa": jepa_val}
    return loss, pred_obs, target_future, stats


def to_device(adapter, embedder, target_ae, device):
    adapter.model.to(device)
    embedder.to(device)
    target_ae.to(device)
    if adapter.CovariateFusion is not None:
        adapter.CovariateFusion.to(device)


def trainable_params(adapter, embedder, target_ae):
    modules = [adapter.model, embedder, target_ae]
    if adapter.CovariateFusion is not None:
        modules.append(adapter.CovariateFusion)
    return [p for m in modules for p in m.parameters() if p.requires_grad]


def set_train_mode(adapter, embedder, target_ae, train: bool):
    adapter.model.train(train)
    embedder.train(train)
    target_ae.train(train)
    if adapter.CovariateFusion is not None:
        adapter.CovariateFusion.train(train)


@torch.no_grad()
def evaluate(adapter, embedder, target_ae, loader, n_latent, n_target, device, max_batches=None):
    set_train_mode(adapter, embedder, target_ae, False)
    abs_sum = sq_sum = count = 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        _, output, target, _ = forward_loss(
            adapter, embedder, target_ae, batch, n_latent, n_target, device, nn.MSELoss())
        abs_sum += (output - target).abs().sum().item()
        sq_sum += ((output - target) ** 2).sum().item()
        count += target.numel()
    set_train_mode(adapter, embedder, target_ae, True)
    if count == 0:
        return {"MAE": float("nan"), "MSE": float("nan"), "RMSE": float("nan")}
    mse = sq_sum / count
    return {"MAE": abs_sum / count, "MSE": mse, "RMSE": mse ** 0.5}
