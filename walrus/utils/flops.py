"""Analytical FLOP accounting for a Walrus model

This file has an analytical formula to count the FLOPs (forward and backwards) of one training microbatch:
    forward_flops: Calculates forward pass FLOPs per microbatch with encoder_proj1/2, decoder_proj1/2, space/time/channel mixing, processor granularity.
    training_step_flops: forward × 3, with gradient-checkpointing recompute reported separately.

Some things to note:
1. The encoder convolves only over the fields the current microbatch dataset has (`proj1.weight[:, field_indices]`, `vstride_encoder.py:349`),
   not over all the field indices the encoder might have. Thus, FLOPS must be recorded per microbatch.
2. Whatever the input data resolution or aspect ratio, the token grid ends up at approximately a fixed count per axis (depending on 2D/3D-ness
   of the data, and boundary conditions) (However, see the next)
3. Microbatch FLOPs are not constant across the data mixture even though the token count might be. Therefore, we must accumulate per microbatch
   using the dataset drawn.
4. Periodic boundaries change the decoder's work. `adaptive_conv_transpose` circularly pads periodic axes before the transposed conv and crops
   afterwards (`vstride_decoder.py:213-238`), so the conv runs on more positions than the token grid implies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from the_well.data.datasets import BoundaryCondition

logger = logging.getLogger(__name__)

# `IsotropicModel._encoder_forward` appends the fixed indices [2, 0, 1]
# (closed_boundary/open_boundary/bias_correction) to state_labels before the encoder,
# and the decoder does not need to reconstruct them.
N_BC_CHANNELS = 3

# Backward is two matmuls per forward matmul
BACKWARD_MULTIPLIER = 2.0

# `FlopAccountant` tracks stages in GRANULAR_FLOP_KEYS individually.
# `forward_flops` reports only forward-pass FLOPs, and then `training_step_flops`
# applies the `(1 + BACKWARD_MULTIPLIER)` rule.

# The keys are a touple becaue `epoch_metrics` all-reduces values packed
# in this order, so every rank must iterate the keys identically.
GRANULAR_FLOP_KEYS = (
    # These add up to the encoder/decoder
    "encoder_proj1",
    "encoder_proj2",
    "decoder_proj1",
    "decoder_proj2",
    # These add up to the processor
    "processor_space_fused_ff_qkv",
    "processor_space_attention",
    "processor_space_attn_out",
    "processor_space_ff_out",
    "processor_time_input_head",
    "processor_time_attention",
    "processor_time_output_head",
    "processor_channel_mixing",
)


def stage_totals(flops: Dict[str, float]) -> Dict[str, float]:
    """Encoder/processor/decoder/total sums of a granular dict (`forward_flops`,
    `training_step_flops`, or accumulated counters). Keys outside GRANULAR_FLOP_KEYS
    (e.g. "useful", "recompute") are ignored.

    Example (forward FLOPs, 200M anchor 768/12/2048 on euler_multi_quadrants_openBC):
        stage_totals(forward_flops(shape, geom))
        -> {"encoder": 197.84e9, "processor": 2747.36e9, "decoder": 169.57e9,
            "total": 3114.76e9}
    """
    totals = {
        stage: sum(flops[k] for k in GRANULAR_FLOP_KEYS if k.startswith(stage))
        for stage in ("encoder", "processor", "decoder")
    }
    totals["total"] = sum(totals.values())
    return totals


def conv_out_size(in_size: int, kernel: int, stride: int) -> int:
    """After a convolution layer slides over a row of grid points along one axis,
    how many output points do we get? A conv's cost grows with how many output points
    it computes, so we need this.

    PyTorch's general formula includes a padding term:
        out = ( ( in + (2 x padding) - kernel ) / stride ) + 1

    But Walrus's `learned_pad` pads the input beforehand in the patch jitterer.
    Thus, `in_size` already includes padding, Thus, the conv is called with `padding=0`
    So we get:
        out = ( ( in - kernel ) / stride ) + 1
        
    A special case is when 2D datasets have a singleton axis, so the general formula
    breaks there. Walrus's adaptive_conv handles this by shrinking window size to 1 on
    such axes so the output is also 1.

    Example (euler_multi_quadrants_openBC, 512 pixels padded to 532, both encoder convs):
        conv_out_size(532, kernel=8, stride=4) -> 132
        conv_out_size(132, kernel=4, stride=4) -> 33     (the token count on that axis)
        conv_out_size(1, kernel=8, stride=1)   -> 1      (singleton axis of 2D data)
    """
    if in_size == 1:
        return 1
    return (in_size - kernel) // stride + 1


def _effective_kernel(kernel: Sequence[int], in_shape: Sequence[int]) -> Tuple[int, ...]:
    """Since all data is treated as 3D, each encoder conv window is a 3D box. So if it is
    8-wide on each axis, we have 8^3 = 512 weights per input-output channel pair, and so
    each output point would cost 512 multiply-adds per input-output channel pair.
    
    2D datasets have a singleton axis. An 8-window sliding over that axis would just sit
    over mostly points that do not exist. `vstride_encoder.py:244-247` does:

        for i, dim in enumerate(spatial_dims[::-1], start=1):
            if dim == 1:                                     # this axis has length 1
                weight = weight.sum(dim=-i, keepdim=True)    # squash the window to 1 along it
                stride[-i] = 1
                padding[-i] = 0
                
    So for the singleton axis of the weight box, it adds the weights and squishes the axis.
    This is because since the data has just one value `x` on that axis, summing the weights
    and multiplying by x is the same as the unsquished weight box axis sliding over x.

    But this costs less. So later when we need the kernel window volume to calculate flops,
    we should use an effective kernel volume with one axis just equal to 1, so we get the
    right cost.

    Example:
        _effective_kernel((8, 8, 8), (532, 532, 1)) -> (8, 8, 1)   2D data: volume 64
        _effective_kernel((8, 8, 8), (148, 148, 148)) -> (8, 8, 8) 3D data: volume 512
    """
    return tuple(1 if s == 1 else k for k, s in zip(kernel, in_shape))


def conv_stage_flops(
        in_shape: Sequence[int],
        kernel: Sequence[int],
        stride: Sequence[int],
        c_in: int,
        c_out: int,
        batch: int,
        transposed: bool = False,
    ) -> int:
    """FLOPs for one adaptive strided (or transposed) conv stage. The encoder uses two
    forward convs, and the decoder can be said to use two transposed convs. One forward
    conv costs one multiply-add per output position, and one transposed conv costs one
    multiply-add per input position.
    
    On periodic axes, the decoder wraps a few extra token points onto each edge before
    the transposed conv and then trims the extra output off afterwards
    (vstride_decoder.py, adaptive_conv_transpose) so we really need to know them.
    `decoder_stage_shapes` works out that padded input size and passes it in as `in_shape`.
    `in_shape` must already include any padding the conv sees.

    Example (200M anchor, euler_multi_quadrants_openBC, T x B = 6 x 2 = 12, 6 fields + 3 BC flags):
        encoder proj1: conv_stage_flops((532, 532, 1), (8, 8, 8), (4, 4, 1),
                                        c_in=9, c_out=352, batch=12)
            = 2 * 12 * (132 * 132 * 1) * (8 * 8 * 1) * 9 * 352  -> 84.79e9
        decoder proj1: conv_stage_flops((33, 33, 1), (4, 4, 4), (4, 4, 1),
                                        c_in=768, c_out=352, batch=12, transposed=True)
            = 2 * 12 * (33 * 33 * 1) * (4 * 4 * 1) * 768 * 352  -> 113.05e9
    """
    eff_kernel = _effective_kernel(kernel, in_shape)
    
    if transposed:
        n_positions = math.prod(in_shape)
    else:
        n_positions = math.prod(
            conv_out_size(s, k, st) for s, k, st in zip(in_shape, eff_kernel, stride)
        )

    return 2 * batch * n_positions * math.prod(eff_kernel) * c_in * c_out


def space_mixing_flops(hidden_dim: int, mlp_dim: int, n_space: int, t_times_b: int) -> Dict[str, int]:
    """One `FullAttention` spatial block, by submodule.

    Example (H=768, M=2048, N=33x33=1089 tokens, T x B = 12):
        space_mixing_flops(768, 2048, 1089, 12)
        -> {"fused_ff_qkv": 87.36e9, "attention": 43.72e9, "attn_out": 15.42e9,
            "ff_out": 20.55e9}
    """
    return {
        "fused_ff_qkv": 2 * t_times_b * n_space * hidden_dim * (mlp_dim + 3 * hidden_dim),
        "attention": 4 * t_times_b * (n_space**2) * hidden_dim,
        "attn_out": 2 * t_times_b * n_space * hidden_dim * hidden_dim,
        "ff_out": 2 * t_times_b * n_space * (mlp_dim // 2) * hidden_dim,
    }


def time_mixing_flops(hidden_dim: int, n_time: int, b_times_n_space: int) -> Dict[str, int]:
    """One `AxialTimeAttention` temporal block, by submodule.

    Example (H=768, T=6 frames, B x N = 2 x 1089 = 2178 spatial locations):
        time_mixing_flops(768, 6, 2178)
        -> {"input_head": 46.25e9, "attention": 0.24e9, "output_head": 15.42e9}
    """
    return {
        "input_head": 2 * (n_time * b_times_n_space) * hidden_dim * (3 * hidden_dim),
        "attention": 4 * b_times_n_space * (n_time**2) * hidden_dim,
        "output_head": 2 * (n_time * b_times_n_space) * hidden_dim * hidden_dim,
    }


def channel_mixing_flops(hidden_dim: int, channel_mlp_dim: Optional[int], n_tokens: int) -> int:
    """0 when channel mixing is `Identity` (the default), else a 2-layer per-token MLP.

    Example (H=768, T x B x N = 12 x 1089 = 13068 tokens):
        channel_mixing_flops(768, None, 13068) -> 0            Identity
        channel_mixing_flops(768, 3072, 13068) -> 123.3e9      MLP of width 3072
    """
    if channel_mlp_dim is None:
        return 0
    return 4 * n_tokens * hidden_dim * channel_mlp_dim


@dataclass(frozen=True)
class ModelShape:
    """Shape knobs and fixed architecture constants that set cost.

    Example (the 200M anchor):
        ModelShape(hidden_dim=768, mlp_dim=2048, processor_blocks=12, intermediate_dim=352,
                   encoder_kernel1=(8, 8, 8), encoder_kernel2=(4, 4, 4),
                   decoder_kernel1=(4, 4, 4), decoder_kernel2=(8, 8, 8),
                   causal_in_time=True, gradient_checkpointing_freq=0, channel_mlp_dim=None)
    """

    hidden_dim: int
    mlp_dim: int
    processor_blocks: int
    intermediate_dim: int
    encoder_kernel1: Tuple[int, ...]
    encoder_kernel2: Tuple[int, ...]
    decoder_kernel1: Tuple[int, ...]
    decoder_kernel2: Tuple[int, ...]
    causal_in_time: bool = False
    gradient_checkpointing_freq: int = 0
    channel_mlp_dim: Optional[int] = None

    @classmethod
    def from_model(cls, model: nn.Module) -> "ModelShape":
        """Read the shape off a live (possibly distribution-wrapped) model."""
        model = unwrap_model(model)
        encoder = next(iter(model.embed.values()))
        decoder = next(iter(model.debed.values()))
        block = model.blocks[0]
        channel_mixing = getattr(block, "channel_mixing", None)
        return cls(
            hidden_dim=model.hidden_dim,
            mlp_dim=block.space_mixing.mlp_dim,
            processor_blocks=len(model.blocks),
            intermediate_dim=encoder.inner_dim,
            encoder_kernel1=tuple(encoder.base_kernel1),
            encoder_kernel2=tuple(encoder.base_kernel2),
            decoder_kernel1=tuple(decoder.base_kernel1),
            decoder_kernel2=tuple(decoder.base_kernel2),
            causal_in_time=model.causal_in_time,
            gradient_checkpointing_freq=model.gradient_checkpointing_freq,
            channel_mlp_dim=(
                None
                if channel_mixing is None or isinstance(channel_mixing, nn.Identity)
                else channel_mixing.fc1.out_features  # walrus.models.shared_utils.mlps.MLP
            ),
        )


@dataclass(frozen=True)
class BatchGeometry:
    """Everything about one micro-batch that affects its FLOP count.

    `encoder_in_shape` is the spatial shape after the patch jitterer's padding and is
    what the encoder's first conv sees. Because this is frozen and hashable, it doubles
    as the cache key in `FlopAccountant`.

    Example (euler_multi_quadrants_openBC: 512x512, open/open, 6 fields, B=2, T=6):
        BatchGeometry(encoder_in_shape=(532, 532, 1), stride1=(4, 4, 1), stride2=(4, 4, 1),
                      periodic=(False, False, False), n_fields_in=9, n_fields_out=6,
                      n_time_in=6, batch_size=2, token_shape=(33, 33, 1),
                      dataset_name="euler_multi_quadrants_openBC")
        .encoder_mid_shape((8, 8, 8))            -> (132, 132, 1)
        .encoder_out_shape((8, 8, 8), (4, 4, 4)) -> (33, 33, 1)
        .n_spatial_tokens                        -> 1089
    """

    encoder_in_shape: Tuple[int, ...]
    stride1: Tuple[int, ...]
    stride2: Tuple[int, ...]
    periodic: Tuple[bool, ...]
    n_fields_in: int
    n_fields_out: int
    n_time_in: int
    batch_size: int
    token_shape: Optional[Tuple[int, ...]] = None
    dataset_name: str = ""

    def encoder_mid_shape(self, kernel1: Sequence[int]) -> Tuple[int, ...]:
        eff = _effective_kernel(kernel1, self.encoder_in_shape)
        return tuple(
            conv_out_size(s, k, st)
            for s, k, st in zip(self.encoder_in_shape, eff, self.stride1)
        )
    
    def encoder_out_shape(self, kernel1, kernel2) -> Tuple[int, ...]:
        mid = self.encoder_mid_shape(kernel1)
        eff = _effective_kernel(kernel2, mid)
        return tuple(
            conv_out_size(s, k, st) for s, k, st in zip(mid, eff, self.stride2)
        )

    @property
    def n_spatial_tokens(self) -> int:
        if self.token_shape is None:
            raise ValueError("token_shape unknown: the encoder's forward hook has not run yet")
        return math.prod(self.token_shape)


def decoder_stage_shapes(
        in_shape: Sequence[int],
        kernel: Sequence[int],
        stride: Sequence[int],
        periodic: Sequence[bool],
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Per axis, for one decoder stage (`adaptive_conv_transpose`, vstride_decoder.py:210-251):
        padded_in: points the transposed conv runs on  -> sets its cost
        out:       points that come out                -> the next stage's input

    The decoder grows the token grid back to pixels in two transposed convs, e.g. for a
    periodic 512-pixel axis: 32 tokens -> 128 -> 516 (the encoder's padded input).

    Output size. Each input point stamps a tile `k` wide; tiles start `s` apart:
        out = (in - 1) * s + k
    k = 4, s = 4: tiles side by side.       k = 8, s = 4: tiles overlap by k - s = 4.

    Periodic axes. The field wraps around, so with overlapping tiles (k > s) the edge
    points also need tiles from points on the opposite end. The decoder therefore:
      1. copies (k - s) // s points from the opposite end onto each side (circular pad);
      2. runs the conv on the padded axis;
      3. crops k - s points off each end (the parts past the real domain).
    With k == s (stage 1) tiles don't overlap: no pad, no crop, same as a non-periodic axis.

    Stage 2 (k = 8, s = 4) on 128 input points:
                   padded_in             conv output             out
        periodic   128 + 2*1 = 130       (130 - 1)*4 + 8 = 524   524 - 2*4 = 516
        open       128                   (128 - 1)*4 + 8 = 516   516
    (an open axis has 33 tokens, so 132 -> 532 there). Periodic costs 130/128 = +1.6 %
    on that stage; that extra work is why `padded_in` is returned.

    Size-1 axes (the padded third axis of 2D data): 1 in, 1 out.

    The round trip back to the encoder's input is exact only when s divides k - s. With
    stride 6 (384-pixel axes, turbulent_radiative_layer_2D) the pad (8-6)//6 = 0 doesn't
    match the crop of 2, so 398 decodes to 394. The model does this too (TODO at
    vstride_decoder.py:225); test_stride_six_axis_does_not_round_trip pins it.

    Example, both stages (strides are the encoder's, swapped):
        periodic 512 axis (encoder input 516, 32 tokens):
            decoder_stage_shapes((32,), (4,), (4,), (True,))  -> ((32,), (128,))
            decoder_stage_shapes((128,), (8,), (4,), (True,)) -> ((130,), (516,))
        open 512 axis (encoder input 532, 33 tokens):
            decoder_stage_shapes((33,), (4,), (4,), (False,))  -> ((33,), (132,))
            decoder_stage_shapes((132,), (8,), (4,), (False,)) -> ((132,), (532,))
    """
    padded, out = [], []
    for size, k, s, is_periodic in zip(in_shape, kernel, stride, periodic):
        if size == 1:
            padded.append(1)
            out.append(1)
            continue
        pad_in = (k - s) // s if is_periodic else 0
        size_padded = size + 2 * pad_in
        size_out = (size_padded - 1) * s + k
        if is_periodic:
            size_out -= 2 * (k - s)
        padded.append(size_padded)
        out.append(size_out)
    return tuple(padded), tuple(out)


def forward_flops(shape: ModelShape, geom: BatchGeometry) -> Dict[str, int]:
    """Forward-pass FLOPs for one micro-batch. Returns per-stage breakdown within
    the encoder, decoder, and processor. The more granular measures can be added to
    provide coarser keys.

    Example (200M anchor, euler_multi_quadrants_openBC geometry above; processor terms are
    per-block values x 12 blocks):
        forward_flops(shape, geom)
        -> {"encoder_proj1": 84.79e9, "encoder_proj2": 113.05e9,
            "decoder_proj1": 113.05e9, "decoder_proj2": 56.52e9,
            "processor_space_fused_ff_qkv": 1048.3e9, "processor_space_attention": 524.6e9,
            "processor_space_attn_out": 185.0e9, "processor_space_ff_out": 246.6e9,
            "processor_time_input_head": 555.0e9, "processor_time_attention": 2.9e9,
            "processor_time_output_head": 185.0e9, "processor_channel_mixing": 0}
        stage_totals(...)["total"] -> 3114.76e9
    """
    tb = geom.n_time_in * geom.batch_size       # Time steps x batch size

    ### Encoder calcs here
    mid_shape = geom.encoder_mid_shape(shape.encoder_kernel1)
    token_shape = geom.token_shape or geom.encoder_out_shape(
        shape.encoder_kernel1, shape.encoder_kernel2
    )
    encoder_proj1 = conv_stage_flops(
        geom.encoder_in_shape,
        shape.encoder_kernel1,
        geom.stride1,
        geom.n_fields_in,
        shape.intermediate_dim,
        batch=tb,
    )
    encoder_proj2 = conv_stage_flops(
        mid_shape,
        shape.encoder_kernel2,
        geom.stride2,
        shape.intermediate_dim,
        shape.hidden_dim,
        batch=tb,
    )

    ### Processor calcs here
    n_space = math.prod(token_shape)            # Number of spatial tokens

    space_breakdown = space_mixing_flops(shape.hidden_dim, shape.mlp_dim, n_space, tb)
    time_breakdown = time_mixing_flops(shape.hidden_dim, geom.n_time_in, geom.batch_size * n_space)
    channel = channel_mixing_flops(shape.hidden_dim, shape.channel_mlp_dim, tb * n_space)

    ### Decoder calcs here

    # The decoder re-expands only the last time step unless the model is causal in time
    # (`isotropic_model.py:397-398`).
    t_out = geom.n_time_in if shape.causal_in_time else 1

    # The decoder swaps the encoder's per-axis stride pair (`vstride_decoder.py:261`).
    dec_stride1, dec_stride2 = geom.stride2, geom.stride1
    padded1, mid1 = decoder_stage_shapes(
        token_shape, shape.decoder_kernel1, dec_stride1, geom.periodic
    )
    padded2, _ = decoder_stage_shapes(
        mid1, shape.decoder_kernel2, dec_stride2, geom.periodic
    )
    decoder_proj1 = conv_stage_flops(
        padded1,
        shape.decoder_kernel1,
        dec_stride1,
        shape.hidden_dim,
        shape.intermediate_dim,
        batch=t_out * geom.batch_size,
        transposed=True,
    )
    decoder_proj2 = conv_stage_flops(
        padded2,
        shape.decoder_kernel2,
        dec_stride2,
        shape.intermediate_dim,
        geom.n_fields_out,
        batch=t_out * geom.batch_size,
        transposed=True,
    )

    return {
        "encoder_proj1": encoder_proj1,
        "encoder_proj2": encoder_proj2,
        "decoder_proj1": decoder_proj1,
        "decoder_proj2": decoder_proj2,
        **{f"processor_space_{k}": shape.processor_blocks * v for k, v in space_breakdown.items()},
        **{f"processor_time_{k}": shape.processor_blocks * v for k, v in time_breakdown.items()},
        "processor_channel_mixing": shape.processor_blocks * channel
    }


def training_step_flops(forward: Dict[str, int], shape: ModelShape) -> Dict[str, float]:
    """Forward + backward FLOPs for one micro-batch, including recompute.

    Gradient checkpointing re-runs part of the forward during backward. That changes
    measured FLOPs without changing useful FLOPs. Two otherwise identical runs with
    different gradient checkpointing frequencies will report different FLOPs, while
    we might be interested in FLOPs as a function of model shape and data only.
    So the recomputing surcharge is reported separately.
    
    The encoder and decoder are checkpointed whenever any checkpointing is on.
    processor block `i` is checkpointed when `i % freq == 0` (`isotropic_model.py:126-135`).

    Returns each granular key as forward FLOPs x `(1 + BACKWARD_MULTIPLIER)` (no
    recompute), plus `useful` (their sum) and the `recompute` surcharge (forward-only).
    Measured FLOPs are `useful + recompute`.

    Example (the forward_flops example: encoder 197.84, processor 2747.36, decoder
    169.57 GF; 12 blocks):
        gradient_checkpointing_freq=0 -> {"useful": 9344.3e9, "recompute": 0.0,
                                          "encoder_proj1": 254.4e9, ...}   (each key x 3)
        gradient_checkpointing_freq=2 -> "recompute": 197.84e9 + 169.57e9
                                          + 2747.36e9 * 6 / 12 = 1741.1e9 (useful unchanged)
    """
    forward_totals = stage_totals(forward)
    freq = shape.gradient_checkpointing_freq
    if freq > 0:
        n_checkpointed = math.ceil(shape.processor_blocks / freq)
        recompute = (
            forward_totals["encoder"]
            + forward_totals["decoder"]
            + forward_totals["processor"] * n_checkpointed / shape.processor_blocks
        )
    else:
        recompute = 0.0

    result = {k: forward[k] * (1.0 + BACKWARD_MULTIPLIER) for k in GRANULAR_FLOP_KEYS}
    result["useful"] = forward_totals["total"] * (1.0 + BACKWARD_MULTIPLIER)
    result["recompute"] = float(recompute)
    return result


def analytic_token_shape(resolution: Sequence[int], periodic: Sequence[bool]) -> Tuple[int, ...]:
    """Token grid Walrus produces for a dataset, without running a model.

    `choose_kernel_size_deterministic` fixes the token count per axis (32 when at most
    two spatial axes are non-singleton, 16 when all three are), and the patch jitterer's
    boundary padding adds exactly one more token on each non-periodic axis. Periodic
    axes get circular padding that cancels.

    Used for planning and unique-token denominator. The accountant itself
    observes the real grid instead.

    Example (checked against the model on all 18 datasets by build_well_registry --verify):
        analytic_token_shape((256, 256, 1), (True, True, False))   -> (32, 32, 1)  active_matter
        analytic_token_shape((512, 128, 1), (True, False, False))  -> (32, 33, 1)  rayleigh_benard
        analytic_token_shape((512, 512, 1), (False, False, False)) -> (33, 33, 1)  euler openBC
        analytic_token_shape((128, 128, 256), (True, True, False)) -> (16, 16, 17) TRL 3D
    """
    non_singleton = sum(1 for r in resolution if r != 1)
    base = 512 // 16 if non_singleton <= 2 else 256 // 16
    return tuple(
        1 if r == 1 else base + (0 if p else 1)
        for r, p in zip(resolution, periodic)
    )


def _dataset_periodic_axes(sub: Any, n_axes: int) -> Tuple[bool, ...]:
    """Per-axis periodicity of a Well sub-dataset, in spatial-axis order.

    Read from the first file with the dataset's own `_padding_bcs`, the reader that fills
    `batch["boundary_conditions"]`. `WellMetadata.boundary_condition_types` can't be used:
    the_well builds it as a *set* of BC types, so it neither says which axis is which nor
    repeats a type (rayleigh_taylor_instability: 3 axes, set {PERIODIC, WALL}).
    Raises if the file can't be read; `unique_model_tokens` then disables the metric.

    Example (rayleigh_benard, 512x128 padded to 3D, periodic x wall):
        _dataset_periodic_axes(sub, 3) -> (True, False, False)
        (the padded singleton third axis defaults to OPEN, so False; it never affects cost)
    """
    import h5py
    from the_well.data.datasets import IO_PARAMS

    with (
        sub.fs.open(sub.files_paths[0], "rb", **IO_PARAMS["fsspec_params"]) as f,
        h5py.File(f, "r", **IO_PARAMS["h5py_params"]) as h5,
    ):
        bcs = sub._padding_bcs(h5, {}, 0, 0, 0, 1)  # cache, sample, t, steps, dt unused
    return _periodic_flags(bcs, n_axes)


def unique_model_tokens(datamodule: Any) -> Optional[int]:
    """`Σ_datasets n_trajectories × n_steps × N_spatial` for the training mixture.

    The denominator of `repetition_factor` (Docs/flop_accounting_wandb.md §6).
    Returns `None` (so `repetition_factor` is not logged) if the datamodule has no
    per-dataset metadata, or if a dataset's per-axis BCs can't be read: no number beats
    a plausible wrong one.

    Example: one 2D dataset, 100 trajectories x 101 steps, fully periodic (32x32 tokens):
        100 * 101 * (32 * 32 * 1) = 10,342,400
    PT9's 9-dataset training mix gives ~3.26e9.
    """
    dataset = getattr(datamodule, "train_dataset", None)
    sub_dsets = getattr(dataset, "sub_dsets", None)
    if not sub_dsets:
        return None

    total = 0
    for sub in sub_dsets:
        meta = sub.metadata
        try:
            periodic = _dataset_periodic_axes(sub, len(meta.spatial_resolution))
        except Exception as e:  # noqa: BLE001 - any read failure disables the metric
            logger.warning(
                "repetition_factor disabled: can't read per-axis BCs of %s (%s)",
                getattr(meta, "dataset_name", "?"),
                e,
            )
            return None
        n_spatial = math.prod(
            analytic_token_shape(meta.spatial_resolution, periodic)
        )
        steps = sum(
            n_traj * n_steps
            for n_traj, n_steps in zip(
                meta.n_trajectories_per_file, meta.n_steps_per_trajectory
            )
        )
        total += steps * n_spatial
    return total


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP/FSDP wrappers to reach the `IsotropicModel` underneath.

    Example: unwrap_model(FSDP(IsotropicModel(...))) -> the IsotropicModel
    (unwrap_model(model) on an unwrapped model returns it unchanged).
    """
    for attr in ("_fsdp_wrapped_module", "module"):
        while hasattr(model, attr) and not hasattr(model, "blocks"):
            model = getattr(model, attr)
    return model


def supports_flop_accounting(model: nn.Module) -> bool:
    """Whether this is an encoder/processor/decoder model the formula describes.

    The baseline wrappers in `walrus/models/baseline_wrappers.py` are not, and run
    through the same `Trainer`.

    Example: IsotropicModel (bare or FSDP-wrapped) -> True; a DPOT/ScOT/MPP wrapper -> False.
    """
    unwrapped = unwrap_model(model)
    return all(
        hasattr(unwrapped, attr) for attr in ("embed", "debed", "blocks", "hidden_dim")
    )


def _periodic_flags(bcs: Any, n_axes: int) -> Tuple[bool, ...]:
    """Per-axis periodicity from a `(n_dims, 2)` boundary-condition tensor.

    Axis order matches the spatial axes, as `adaptive_conv_transpose` assumes. Axes
    beyond the stored BCs (or no BCs at all) count as non-periodic. Each row is
    (low side, high side); the_well's codes are WALL=0, OPEN=1, PERIODIC=2.

    Example (rayleigh_benard padded to 3D: periodic x, wall y, padded z):
        _periodic_flags(torch.tensor([[2, 2], [0, 0], [1, 1]]), 3) -> (True, False, False)
        _periodic_flags(torch.tensor([[2, 2], [0, 0]]), 3)         -> (True, False, False)
        _periodic_flags(None, 3)                                   -> (False, False, False)
    """
    # One `.tolist()`, not per-element tensor indexing: this runs in the encoder pre-hook
    # every micro-batch, and indexing cost ~200 us there vs ~5 us.
    rows = [] if bcs is None else (bcs.tolist() if hasattr(bcs, "tolist") else bcs)
    flags = [int(r[0]) == BoundaryCondition.PERIODIC.value for r in rows[:n_axes]]
    return tuple(flags + [False] * (n_axes - len(flags)))


class FlopAccountant:
    """Accumulates per-micro-batch FLOPs and token counts over a training epoch.

    Geometry is observed from the model rather than re-derived from the input
    resolution: a forward pre-hook on the encoder records the padded input shape, the
    strides actually chosen, the boundary conditions, and the real per-dataset channel
    count, and a forward hook records the resulting token grid. Everything downstream is
    exact arithmetic on those observations. The FLOP evaluation itself is cached on the
    geometry, so only the first micro-batch of each `(dataset, resolution, shape)`
    combination costs anything.

    Counts are per rank. `max_samples` caps micro-batches per rank, so global compute is
    `world_size x` the per-rank figure; `epoch_metrics` reports both, and reduces across
    ranks exactly when a process group exists.
    """

    def __init__(
        self,
        model: nn.Module,
        world_size: int = 1,
        unique_model_tokens: Optional[int] = None,
    ):
        self.shape = ModelShape.from_model(model)
        self.world_size = world_size
        self.unique_model_tokens = unique_model_tokens
        self._cache: Dict[BatchGeometry, Dict[str, float]] = {}
        # Last *complete* geometry (token grid included): what record_microbatch prices.
        self._observed: Optional[BatchGeometry] = None
        # Geometry seen by the pre-hook, committed to _observed only by the post-hook.
        # Gradient checkpointing re-runs the encoder during backward, and non-reentrant
        # checkpointing may stop that recompute before the encoder returns, so the
        # pre-hook can fire without its post-hook. Writing straight into _observed then
        # left it without a token grid, and record_microbatch raised.
        self._pending: Optional[BatchGeometry] = None
        self._handles: list = []
        self.cumulative_granular_flops = {k: 0.0 for k in GRANULAR_FLOP_KEYS}
        self.cumulative_recompute_flops = 0.0
        self.cumulative_tokens = 0
        self.reset_epoch()
        self._attach(model)

    def _attach(self, model: nn.Module) -> None:
        unwrapped = unwrap_model(model)
        for encoder in unwrapped.embed.values():
            self._handles.append(
                encoder.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
            )
            self._handles.append(
                encoder.register_forward_hook(self._post_hook, with_kwargs=False)
            )

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def state_dict(self) -> Dict[str, Any]:
        """The run-long counters, for checkpoint metadata.

        Everything else is per-epoch or a cache. The counters are all-reduced in
        `epoch_metrics`, so they are identical on every rank.
        """
        return {
            "cumulative_granular_flops": dict(self.cumulative_granular_flops),
            "cumulative_recompute_flops": self.cumulative_recompute_flops,
            "cumulative_tokens": self.cumulative_tokens,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the run-long counters on resume; absent keys stay at 0."""
        granular = state.get("cumulative_granular_flops", {})
        self.cumulative_granular_flops = {
            k: float(granular.get(k, 0.0)) for k in GRANULAR_FLOP_KEYS
        }
        self.cumulative_recompute_flops = float(state.get("cumulative_recompute_flops", 0.0))
        self.cumulative_tokens = int(state.get("cumulative_tokens", 0))

    def _pre_hook(self, module, args, kwargs):
        # SpaceBagAdaptiveDVstrideEncoder.forward(x, field_indices, bcs, metadata, *,
        # random_kernel). `field_indices` here is state_labels with the 3 BC-flag
        # indices already appended by IsotropicModel._encoder_forward.
        x, field_indices = args[0], args[1]
        bcs = args[2] if len(args) > 2 else kwargs.get("bcs")
        random_kernel = kwargs.get("random_kernel")
        if random_kernel is None:
            return
        spatial = tuple(int(s) for s in x.shape[3:])
        n_axes = len(spatial)
        n_fields_in = int(field_indices.numel())
        self._pending = BatchGeometry(
            encoder_in_shape=spatial,
            stride1=tuple(int(random_kernel[i][0]) for i in range(n_axes)),
            stride2=tuple(int(random_kernel[i][1]) for i in range(n_axes)),
            periodic=_periodic_flags(bcs, n_axes),
            n_fields_in=n_fields_in,
            n_fields_out=n_fields_in - N_BC_CHANNELS,
            n_time_in=int(x.shape[0]),
            batch_size=int(x.shape[1]),
        )

    def _post_hook(self, module, args, output):
        if self._pending is None:
            return
        x_out = output[0] if isinstance(output, tuple) else output
        token_shape = tuple(int(s) for s in x_out.shape[3:])
        self._observed = BatchGeometry(
            **{**self._pending.__dict__, "token_shape": token_shape}
        )
        self._pending = None

    def reset_epoch(self) -> None:
        self._epoch_granular_flops = {k: 0.0 for k in GRANULAR_FLOP_KEYS}
        self._epoch_recompute_flops = 0.0
        self._epoch_tokens = 0
        self._epoch_microbatches = 0

    def record_microbatch(self, dataset_name: str = "") -> Optional[Dict[str, float]]:
        """Account for the micro-batch that was just run through the model.

        Returns `None` if no forward pass has been observed yet (e.g. a fake pass in
        distributed validation), so callers can treat that as a no-op.
        """
        geom = self._observed
        if geom is None:
            return None
        if dataset_name:
            geom = BatchGeometry(**{**geom.__dict__, "dataset_name": dataset_name})

        entry = self._cache.get(geom)
        if entry is None:
            # Granular keys (fwd+bwd), "useful", "recompute", plus the token count.
            entry = training_step_flops(forward_flops(self.shape, geom), self.shape)
            entry["tokens"] = float(geom.batch_size * geom.n_time_in * geom.n_spatial_tokens)
            self._cache[geom] = entry
            logger.debug(
                "FLOP geometry %s: %.3f GF/micro-batch fwd+bwd",
                dataset_name or geom.encoder_in_shape,
                entry["useful"] / 1e9,
            )

        for k in GRANULAR_FLOP_KEYS:
            self._epoch_granular_flops[k] += entry[k]
        self._epoch_recompute_flops += entry["recompute"]
        self._epoch_tokens += int(entry["tokens"])
        self._epoch_microbatches += 1
        return entry

    def epoch_metrics(self, elapsed_seconds: float) -> Dict[str, float]:
        """Per-epoch compute metrics for the training log (keys: Docs/flop_accounting_wandb.md).

        Must be called on every rank: it all-reduces when a process group exists, so the
        totals are exact rather than `rank 0 x world_size`.
        """
        local = torch.tensor(
            [
                *(self._epoch_granular_flops[k] for k in GRANULAR_FLOP_KEYS),
                self._epoch_recompute_flops,
                float(self._epoch_tokens),
                float(self._epoch_microbatches),
            ],
            dtype=torch.float64,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            total = local.clone().to(_reduce_device())
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
            total = total.cpu()
        else:
            total = local * self.world_size

        *granular_values, recompute, tokens, microbatches = total.tolist()
        granular = dict(zip(GRANULAR_FLOP_KEYS, granular_values))
        for k in GRANULAR_FLOP_KEYS:
            self.cumulative_granular_flops[k] += granular[k]
        self.cumulative_recompute_flops += recompute
        self.cumulative_tokens += int(tokens)

        # "flops" means useful fwd+bwd FLOPs (no recompute); measured = flops + recompute.
        epoch = stage_totals(granular)
        cumulative = stage_totals(self.cumulative_granular_flops)
        metrics = {
            "epoch_flops": epoch["total"],
            "cumulative_flops": cumulative["total"],
            "epoch_recompute_flops": recompute,
            "cumulative_recompute_flops": self.cumulative_recompute_flops,
            "cumulative_processor_flops": cumulative["processor"],
            "flops_per_microbatch": epoch["total"] / microbatches if microbatches else 0.0,
            # Every granular key is forward x 3, so this equals the forward-only fraction.
            "enc_dec_flop_fraction": (
                (epoch["encoder"] + epoch["decoder"]) / epoch["total"] if epoch["total"] else 0.0
            ),
            "tokens_seen": float(self.cumulative_tokens),
            "gpu_hours": elapsed_seconds * self.world_size / 3600.0,
        }
        for k in GRANULAR_FLOP_KEYS:
            metrics[f"epoch_{k}_flops"] = granular[k]
            metrics[f"cumulative_{k}_flops"] = self.cumulative_granular_flops[k]
        if self.unique_model_tokens:
            metrics["repetition_factor"] = (
                self.cumulative_tokens / self.unique_model_tokens
            )
        return metrics


def _reduce_device() -> torch.device:
    """NCCL cannot reduce CPU tensors; Gloo can."""
    backend = torch.distributed.get_backend()
    if backend == "nccl" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")
