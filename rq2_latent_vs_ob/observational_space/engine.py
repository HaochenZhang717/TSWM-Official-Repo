from __future__ import annotations

import torch
import torch.nn as nn


def assemble_batch(batch, embedder, n_target, label_len, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    th, tf = batch["target_history"], batch["target_future"]
    L, H = th.shape[1], tf.shape[1]

    cov_hist = embedder.build_covariate(
        batch["continuous_history"], batch["categorical_history"], batch["exog_history"])
    cov_fut = embedder.build_covariate(
        batch["continuous_future"], batch["categorical_future"], batch["exog_future"])

    model_input = torch.cat([th, cov_hist], dim=-1)

    B = th.shape[0]
    series_dec = th.new_zeros((B, label_len + H, n_target))
    series_dec[:, :label_len] = th[:, L - label_len:]
    cov_full = torch.cat([cov_hist, cov_fut], dim=1)
    cov_dec = cov_full[:, L - label_len:]
    target = torch.cat([series_dec, cov_dec], dim=-1)

    mark_full = torch.cat([batch["mark_history"], batch["mark_future"]], dim=1)
    mark_enc = batch["mark_history"]
    mark_dec = mark_full[:, L - label_len:]
    return model_input, target, mark_enc, mark_dec, cov_fut, tf


def forward_loss(adapter, embedder, batch, n_target, device, criterion):
    label_len = adapter.config.label_len
    model_input, target, mark_enc, mark_dec, exog_future, target_future = assemble_batch(
        batch, embedder, n_target, label_len, device)

    out = adapter._process(model_input, target, mark_enc, mark_dec, exog_future)
    output = out["output"][:, -target_future.shape[1]:, :n_target]
    if adapter.CovariateFusion is not None:
        output = adapter.CovariateFusion(exog_future, output)

    main = criterion(output, target_future)
    loss = main
    aux_val = 0.0
    extra = out.get("additional_loss")
    if extra is not None and torch.is_tensor(extra):
        aux = extra.mean()
        loss = main + aux
        aux_val = aux.item()
    stats = {"main": main.item(), "aux": aux_val}
    return loss, output, target_future, stats


def to_device(adapter, embedder, device):
    adapter.model.to(device)
    embedder.to(device)
    if adapter.CovariateFusion is not None:
        adapter.CovariateFusion.to(device)


def trainable_params(adapter, embedder):
    params = list(adapter.model.parameters()) + list(embedder.parameters())
    if adapter.CovariateFusion is not None:
        params += list(adapter.CovariateFusion.parameters())
    return params


def set_train_mode(adapter, embedder, train: bool):
    adapter.model.train(train)
    embedder.train(train)
    if adapter.CovariateFusion is not None:
        adapter.CovariateFusion.train(train)


@torch.no_grad()
def evaluate(adapter, embedder, loader, n_target, device, max_batches=None):
    set_train_mode(adapter, embedder, False)
    abs_sum = sq_sum = count = 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        _, output, target, _ = forward_loss(adapter, embedder, batch, n_target, device, nn.MSELoss())
        abs_sum += (output - target).abs().sum().item()
        sq_sum += ((output - target) ** 2).sum().item()
        count += target.numel()
    set_train_mode(adapter, embedder, True)
    if count == 0:
        return {"MAE": float("nan"), "MSE": float("nan"), "RMSE": float("nan")}
    mse = sq_sum / count
    return {"MAE": abs_sum / count, "MSE": mse, "RMSE": mse ** 0.5}
