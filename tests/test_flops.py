import os.path as osp
import pathlib
import sys

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.utils.flop_counter import FlopCounterMode

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.models.shared_utils.flexi_utils import choose_kernel_size_deterministic
from walrus.train import CONFIG_DIR, CONFIG_NAME
from walrus.utils.flops import (
    BACKWARD_MULTIPLIER,
    GRANULAR_FLOP_KEYS,
    BatchGeometry,
    FlopAccountant,
    ModelShape,
    analytic_token_shape,
    conv_out_size,
    decoder_stage_shapes,
    forward_flops,
    stage_totals,
    supports_flop_accounting,
    training_step_flops,
)

# The planner lives outside the package; its scripts import each other by bare name.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "Tools" / "flop_planner"))
from well_registry import encoder_shapes_from_tokens  # type: ignore[import-not-found]

ENCODER_KERNEL1 = (8, 8, 8)
ENCODER_KERNEL2 = (4, 4, 4)
# The decoder reverses the roles of the two base kernels (vstride_decoder.py __init__).
DECODER_KERNEL1 = ENCODER_KERNEL2
DECODER_KERNEL2 = ENCODER_KERNEL1


# (resolution, boundary conditions, token grid) as measured by running the model,
# walrus_tokens_and_data_pipeline.md 1.2.
@pytest.mark.parametrize(
    "resolution,periodic,expected",
    [
        ((256, 256, 1), (True, True, True), (32, 32, 1)),
        ((512, 512, 1), (True, True, True), (32, 32, 1)),
        ((128, 384, 1), (True, True, True), (32, 32, 1)),
        ((256, 512, 1), (True, True, True), (32, 32, 1)),
        ((256, 256, 1), (False, False, True), (33, 33, 1)),
        ((1024, 256, 1), (False, False, True), (33, 33, 1)),
        # Mixed boundaries add a token only on the non-periodic axis.
        ((512, 512, 1), (True, False, True), (32, 33, 1)),
        ((64, 64, 64), (True, True, True), (16, 16, 16)),
    ],
)
def test_analytic_token_shape(resolution, periodic, expected):
    assert analytic_token_shape(resolution, periodic) == expected


@pytest.mark.parametrize("axis_resolution", [128, 256, 512, 1024])
@pytest.mark.parametrize("is_periodic", [True, False])
def test_decoder_inverts_encoder(axis_resolution, is_periodic):
    """The decoder must land back exactly on the encoder's padded input grid.

    This is the invariant that pins `decoder_stage_shapes`: the circular pre-pad on
    periodic axes and the post-crop have to cancel, stage by stage, or the model could
    not reconstruct the field at its input resolution. 384-pixel axes are excluded
    because the model itself does not satisfy it there — see the test below.
    """
    (stride1, stride2) = choose_kernel_size_deterministic((axis_resolution,))[0]
    tokens = 32 + (0 if is_periodic else 1)

    padded_in, mid = encoder_shapes_from_tokens(
        (tokens,), (stride1,), (stride2,), ENCODER_KERNEL1[:1], ENCODER_KERNEL2[:1]
    )
    # Forward: the encoder's two padding=0 convs reach the token grid.
    assert conv_out_size(padded_in[0], ENCODER_KERNEL1[0], stride1) == mid[0]
    assert conv_out_size(mid[0], ENCODER_KERNEL2[0], stride2) == tokens

    # Backward: the decoder swaps the stride pair and must retrace those two shapes.
    _, dec_mid = decoder_stage_shapes(
        (tokens,), DECODER_KERNEL1[:1], (stride2,), (is_periodic,)
    )
    _, dec_out = decoder_stage_shapes(
        dec_mid, DECODER_KERNEL2[:1], (stride1,), (is_periodic,)
    )
    assert dec_mid == mid
    assert dec_out == padded_in


def test_stride_six_axis_does_not_round_trip():
    """A 384-pixel periodic axis decodes to 394, not the 398 it was encoded from.

    `adaptive_conv_transpose` pads by `(k - s) // s` but crops by `k - s`, so the two
    only cancel when `s` divides `k - s`. Stride 6 (from `patch_dict[12]`, i.e. any
    384-pixel axis — `turbulent_radiative_layer_2D` has one) leaves `(8 - 6) // 6 == 0`
    padding against a 2-wide crop per side. This is the case the source flags with
    "TODO (mm) - This might get messed up for future non-divisible cases"
    ([vstride_decoder.py:225](../walrus/models/decoders/vstride_decoder.py#L225)).

    It is asserted here so the accountant's numbers stay pinned to what the model does
    rather than to what the geometry ought to be: the FLOP count is verified exact
    against FlopCounterMode on that dataset regardless.
    """
    padded_in, mid = encoder_shapes_from_tokens(
        (32,), (6,), (2,), ENCODER_KERNEL1[:1], ENCODER_KERNEL2[:1]
    )
    assert (padded_in, mid) == ((398,), (66,))

    _, dec_mid = decoder_stage_shapes((32,), DECODER_KERNEL1[:1], (2,), (True,))
    _, dec_out = decoder_stage_shapes(dec_mid, DECODER_KERNEL2[:1], (6,), (True,))
    assert dec_mid == mid
    assert dec_out == (394,)


def test_periodic_axes_cost_more_than_the_token_grid_implies():
    """Circular padding makes the transposed conv run on extra positions.

    Missing this term is what made an earlier hand-derivation undercount the decoder on
    periodic datasets while matching exactly on non-periodic ones.
    """
    periodic_padded, _ = decoder_stage_shapes((32,), (4,), (2,), (True,))
    open_padded, _ = decoder_stage_shapes((32,), (4,), (2,), (False,))
    assert periodic_padded == (34,)
    assert open_padded == (32,)


def test_gradient_checkpointing_surcharge():
    shape = ModelShape(
        hidden_dim=64,
        mlp_dim=256,
        processor_blocks=4,
        intermediate_dim=32,
        encoder_kernel1=ENCODER_KERNEL1,
        encoder_kernel2=ENCODER_KERNEL2,
        decoder_kernel1=DECODER_KERNEL1,
        decoder_kernel2=DECODER_KERNEL2,
        gradient_checkpointing_freq=0,
    )
    # encoder 10, processor 100, decoder 20: forward total 130.
    forward = {k: 0.0 for k in GRANULAR_FLOP_KEYS} | {
        "encoder_proj1": 6.0,
        "encoder_proj2": 4.0,
        "processor_space_attention": 60.0,
        "processor_time_attention": 40.0,
        "decoder_proj1": 8.0,
        "decoder_proj2": 12.0,
    }

    off = training_step_flops(forward, shape)
    assert off["recompute"] == 0
    assert off["useful"] == pytest.approx(130.0 * (1 + BACKWARD_MULTIPLIER))
    # Every granular key is forward*(1+BACKWARD_MULTIPLIER), no recompute folded in.
    for k in GRANULAR_FLOP_KEYS:
        assert off[k] == pytest.approx(forward[k] * (1 + BACKWARD_MULTIPLIER))

    # freq=1 checkpoints every block, so the whole model is recomputed once.
    every = training_step_flops(
        forward, ModelShape(**{**shape.__dict__, "gradient_checkpointing_freq": 1})
    )
    assert every["recompute"] == pytest.approx(130.0)

    # freq=2 checkpoints half the blocks; encoder and decoder are always recomputed.
    half = training_step_flops(
        forward, ModelShape(**{**shape.__dict__, "gradient_checkpointing_freq": 2})
    )
    assert half["recompute"] == pytest.approx(10.0 + 20.0 + 50.0)


def test_unsupported_models_are_detected():
    """Baseline wrappers run through the same Trainer but have no encoder/processor/
    decoder structure, so they must be recognised as unaccountable rather than crash."""

    class NotIsotropic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)

    assert not supports_flop_accounting(NotIsotropic())


def _measured_subtrees(counter):
    """FlopCounterMode totals for the encoder and decoder subtrees."""
    encoder = decoder = 0
    for module, ops in counter.get_flop_counts().items():
        parts = module.split(".")
        if len(parts) != 3:
            continue
        if parts[1] == "embed":
            encoder += sum(ops.values())
        elif parts[1] == "debed":
            decoder += sum(ops.values())
    return encoder, decoder


def test_encoder_decoder_flops_match_flop_counter(default_conf):
    """The closed form must equal FlopCounterMode exactly on a real forward pass.

    Only the encoder and decoder are compared: they are convolutions, which are
    registered ops on a CPU build, whereas `FullAttention`'s spatial attention dispatches
    to a CPU SDPA kernel that has no registered formula and is silently counted as zero
    (walrus_flops_analysis.ipynb 4). The processor terms are verified on GPU instead, by
    LeonardoRunScripts/flop_counter_backend_test.py.
    """
    overrides = [f"{key}={val}" for key, val in default_conf.items()]
    overrides.append("model/processor/space_mixing=full_spatial_attention")
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)

    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        datamodule = instantiate(
            cfg.data.module_parameters,
            world_size=1,
            rank=0,
            data_workers=cfg.data_workers,
            well_base_path=cfg.data.well_base_path,
            field_index_map_override={},
            transform=None,
        )
        n_states = max(datamodule.train_dataset.field_to_index_map.values()) + 1
        model = instantiate(cfg.model, n_states=n_states)
        model.eval()

        batch = datamodule.train_dataset[0]
        inputs, _ = ChannelsFirstWithTimeFormatter().process_input(
            batch,
            causal_in_time=cfg.model.causal_in_time,
            predict_delta=cfg.trainer.prediction_type == "delta",
            train=True,
        )

    accountant = FlopAccountant(model, world_size=1)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(inputs[0], inputs[1], inputs[2].tolist(), metadata=batch["metadata"])

    geometry = accountant._observed
    assert geometry is not None, "encoder hook never fired"
    predicted = stage_totals(forward_flops(accountant.shape, geometry))
    measured_encoder, measured_decoder = _measured_subtrees(counter)

    assert predicted["encoder"] == measured_encoder
    assert predicted["decoder"] == measured_decoder


def test_accountant_accumulates_per_microbatch(default_conf):
    """Totals must come from the dataset actually drawn, not shape x step count."""
    overrides = [f"{key}={val}" for key, val in default_conf.items()]
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)

    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        datamodule = instantiate(
            cfg.data.module_parameters,
            world_size=1,
            rank=0,
            data_workers=cfg.data_workers,
            well_base_path=cfg.data.well_base_path,
            field_index_map_override={},
            transform=None,
        )
        n_states = max(datamodule.train_dataset.field_to_index_map.values()) + 1
        model = instantiate(cfg.model, n_states=n_states)
        model.eval()
        batch = datamodule.train_dataset[0]
        inputs, _ = ChannelsFirstWithTimeFormatter().process_input(
            batch,
            causal_in_time=cfg.model.causal_in_time,
            predict_delta=cfg.trainer.prediction_type == "delta",
            train=True,
        )

    accountant = FlopAccountant(model, world_size=1)
    with torch.no_grad():
        for _ in range(3):
            model(inputs[0], inputs[1], inputs[2].tolist(), metadata=batch["metadata"])
            accountant.record_microbatch("dummy")

    metrics = accountant.epoch_metrics(elapsed_seconds=60.0)
    assert metrics["epoch_flops"] > 0
    assert metrics["epoch_recompute_flops"] == 0  # no gradient checkpointing here
    assert metrics["flops_per_microbatch"] == pytest.approx(metrics["epoch_flops"] / 3)
    assert metrics["cumulative_flops"] == pytest.approx(metrics["epoch_flops"])
    assert 0 < metrics["enc_dec_flop_fraction"] <= 1
    assert metrics["tokens_seen"] > 0
    assert metrics["gpu_hours"] == pytest.approx(60.0 / 3600.0)

    # A second epoch must add to, not replace, the running totals.
    accountant.reset_epoch()
    with torch.no_grad():
        model(inputs[0], inputs[1], inputs[2].tolist(), metadata=batch["metadata"])
        accountant.record_microbatch("dummy")
    second = accountant.epoch_metrics(elapsed_seconds=60.0)
    assert second["cumulative_flops"] > metrics["cumulative_flops"]

    # Run-long counters round-trip through state_dict (checkpoint metadata on resume),
    # and a fresh accountant continues from them; missing keys mean "start from 0".
    state = accountant.state_dict()
    accountant.detach()
    resumed = FlopAccountant(model, world_size=1)
    resumed.load_state_dict(state)
    assert resumed.state_dict() == state
    resumed.load_state_dict({})
    assert stage_totals(resumed.cumulative_granular_flops)["total"] == 0.0
    assert resumed.cumulative_recompute_flops == 0.0 and resumed.cumulative_tokens == 0
    resumed.detach()


def test_accountant_reports_granular_flops(default_conf):
    """Every granular key must reach `epoch_metrics` (what the trainer logs to W&B) as
    forward x (1 + BACKWARD_MULTIPLIER), and the coarse metrics must be their sums."""
    overrides = [f"{key}={val}" for key, val in default_conf.items()]
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)

    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        datamodule = instantiate(
            cfg.data.module_parameters,
            world_size=1,
            rank=0,
            data_workers=cfg.data_workers,
            well_base_path=cfg.data.well_base_path,
            field_index_map_override={},
            transform=None,
        )
        n_states = max(datamodule.train_dataset.field_to_index_map.values()) + 1
        model = instantiate(cfg.model, n_states=n_states)
        model.eval()
        batch = datamodule.train_dataset[0]
        inputs, _ = ChannelsFirstWithTimeFormatter().process_input(
            batch,
            causal_in_time=cfg.model.causal_in_time,
            predict_delta=cfg.trainer.prediction_type == "delta",
            train=True,
        )

    accountant = FlopAccountant(model, world_size=1)
    entries = []
    with torch.no_grad():
        for _ in range(2):
            model(inputs[0], inputs[1], inputs[2].tolist(), metadata=batch["metadata"])
            entries.append(accountant.record_microbatch("dummy"))

    metrics = accountant.epoch_metrics(elapsed_seconds=60.0)
    # Both micro-batches have the same geometry, so each logged key is 2 x one entry.
    forward = forward_flops(accountant.shape, accountant._observed)
    for k in GRANULAR_FLOP_KEYS:
        assert metrics[f"epoch_{k}_flops"] == pytest.approx(sum(e[k] for e in entries))
        assert metrics[f"epoch_{k}_flops"] == pytest.approx(
            2 * forward[k] * (1.0 + BACKWARD_MULTIPLIER)
        )
        assert metrics[f"cumulative_{k}_flops"] == pytest.approx(metrics[f"epoch_{k}_flops"])

    # Coarse metrics are sums of the granular ones.
    epoch = stage_totals({k: metrics[f"epoch_{k}_flops"] for k in GRANULAR_FLOP_KEYS})
    assert metrics["epoch_flops"] == pytest.approx(epoch["total"]) and epoch["total"] > 0
    assert metrics["cumulative_processor_flops"] == pytest.approx(epoch["processor"])
    fwd = stage_totals(forward)
    assert metrics["enc_dec_flop_fraction"] == pytest.approx(
        (fwd["encoder"] + fwd["decoder"]) / fwd["total"]
    )

    # A second epoch must add to, not replace, the cumulative totals.
    accountant.reset_epoch()
    with torch.no_grad():
        model(inputs[0], inputs[1], inputs[2].tolist(), metadata=batch["metadata"])
        accountant.record_microbatch("dummy")
    second = accountant.epoch_metrics(elapsed_seconds=60.0)
    assert second["cumulative_flops"] == pytest.approx(1.5 * metrics["cumulative_flops"])


def test_accountant_survives_gradient_checkpointing_backward(default_conf):
    """With gradient checkpointing, backward re-runs the encoder forward and may stop that
    recompute before the encoder returns, so its pre-hook fires without its post-hook.
    record_microbatch must still price the complete geometry of the training forward.
    (Before the fix this raised "token_shape unknown" on the first micro-batch.)"""
    overrides = [f"{key}={val}" for key, val in default_conf.items()]
    overrides += ["model.gradient_checkpointing_freq=1"]
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)

    with initialize(config_path=str(cfg_dir)):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        datamodule = instantiate(
            cfg.data.module_parameters,
            world_size=1,
            rank=0,
            data_workers=cfg.data_workers,
            well_base_path=cfg.data.well_base_path,
            field_index_map_override={},
            transform=None,
        )
        n_states = max(datamodule.train_dataset.field_to_index_map.values()) + 1
        model = instantiate(cfg.model, n_states=n_states)
        model.train()
        batch = datamodule.train_dataset[0]
        inputs, _ = ChannelsFirstWithTimeFormatter().process_input(
            batch,
            causal_in_time=cfg.model.causal_in_time,
            predict_delta=cfg.trainer.prediction_type == "delta",
            train=True,
        )

    accountant = FlopAccountant(model, world_size=1)
    assert accountant.shape.gradient_checkpointing_freq == 1
    for _ in range(2):
        out = model(inputs[0], inputs[1], inputs[2].tolist(), metadata=batch["metadata"])
        out.float().pow(2).mean().backward()
        entry = accountant.record_microbatch("dummy")
        assert entry is not None and entry["tokens"] > 0

    metrics = accountant.epoch_metrics(elapsed_seconds=1.0)
    forward = forward_flops(accountant.shape, accountant._observed)
    step = training_step_flops(forward, accountant.shape)
    assert step["recompute"] > 0
    # "flops" is useful only; recompute is logged on its own.
    assert metrics["epoch_flops"] == pytest.approx(2 * step["useful"])
    assert metrics["epoch_recompute_flops"] == pytest.approx(2 * step["recompute"])
    accountant.detach()


def test_granular_keys_add_up():
    """forward_flops returns every granular key, and training_step_flops' useful and
    recompute follow from their stage sums. Channel MLP on, so every key is non-zero."""
    shape = ModelShape(
        hidden_dim=64,
        mlp_dim=256,
        processor_blocks=4,
        intermediate_dim=16,
        encoder_kernel1=ENCODER_KERNEL1,
        encoder_kernel2=ENCODER_KERNEL2,
        decoder_kernel1=DECODER_KERNEL1,
        decoder_kernel2=DECODER_KERNEL2,
        gradient_checkpointing_freq=2,
        channel_mlp_dim=128,
    )
    geom = BatchGeometry(
        encoder_in_shape=(264, 256, 1),
        stride1=(8, 8, 1),
        stride2=(1, 1, 1),
        periodic=(False, True, False),
        n_fields_in=7,
        n_fields_out=4,
        n_time_in=3,
        batch_size=2,
    )
    forward = forward_flops(shape, geom)
    assert set(forward) == set(GRANULAR_FLOP_KEYS)
    assert all(forward[k] > 0 for k in GRANULAR_FLOP_KEYS)

    fwd = stage_totals(forward)
    assert fwd["total"] == fwd["encoder"] + fwd["processor"] + fwd["decoder"]
    step = training_step_flops(forward, shape)
    assert step["useful"] == pytest.approx(fwd["total"] * (1 + BACKWARD_MULTIPLIER))
    # freq=2 of 4 blocks: encoder + decoder + half the processor.
    assert step["recompute"] == pytest.approx(fwd["encoder"] + fwd["decoder"] + fwd["processor"] / 2)
    assert stage_totals(step)["total"] == pytest.approx(step["useful"])
