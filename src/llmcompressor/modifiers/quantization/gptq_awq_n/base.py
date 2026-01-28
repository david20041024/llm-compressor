import contextlib
from typing import Dict, List, Optional, Tuple, Union

import torch
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_args import ActivationOrdering
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    getattr_chain,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import PrivateAttr

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_hessian,
    quantize_weight,
)
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.sentinel import Sentinel
from llmcompressor.utils.metric_logging import CompressionLogger

__all__ = ["GPTQAWQModifier"]


class GPTQAWQModifier(Modifier, QuantizationMixin):
    """
    Implements the GPTQ algorithm with AWQ duo-scaling enhancement.
    
    Enhanced features:
    1. AWQ duo-scaling: Uses both output activation and weight statistics for scaling
    2. Direct ratio=0.5: No grid search, uses balanced scaling
    3. Compatible with GPTQ Hessian accumulation

    Sample yaml:

    ```yaml
    test_stage:
      obcq_modifiers:
        GPTQModifier:
          block_size: 128
          dampening_frac: 0.001
          offload_hessians: False
          actorder: static
          duo_scaling: true  # Enable AWQ duo-scaling
          config_groups:
            group_0:
              targets:
                - "Linear"
              weights:
                num_bits: 4
                type: "int"
                symmetric: false
                strategy: group
                group_size: 128
    ```
    """

    # AWQ duo-scaling enhancement
    duo_scaling: bool = True  # 是否啟用AWQ duo-scaling
    
    # gptq modifier arguments
    sequential_targets: Union[str, List[str], None] = None
    block_size: int = 128
    dampening_frac: Optional[float] = 0.01
    actorder: Optional[Union[ActivationOrdering, Sentinel]] = Sentinel("static")
    offload_hessians: bool = False

    # private variables
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)

    # AWQ duo-scaling related private variables
    _activation_means: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _activation_counts: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)

    def resolve_quantization_config(self) -> QuantizationConfig:
        config = super().resolve_quantization_config()

        def resolve_actorder(existing):
            # sentinel default only overrides if existing is None
            if self.actorder == Sentinel("static"):
                return ActivationOrdering.STATIC if existing is None else existing

            # user-provided value always attempts to override
            if existing is None or self.actorder == existing:
                return self.actorder

            # if existing provided and conflicts
            raise ValueError(
                "Cannot resolve activation ordering when both "
                "`GPTQModifier.actorder` and `QuantizationScheme.actorder` "
                f"are provided and differ ({self.actorder}, {existing}). "
                "Either unset `GPTQModifier.actorder` or "
                "remove `actorder` from config groups."
            )

        for scheme in config.config_groups.values():
            assert isinstance(scheme, QuantizationScheme)
            if (
                getattr_chain(scheme, "weights.strategy", None)
                == QuantizationStrategy.GROUP
            ):
                scheme.weights.actorder = resolve_actorder(scheme.weights.actorder)
        return config

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run the GPTQ algorithm with AWQ enhancement
        """
        # apply config to model and prepare calibration hooks
        if QuantizationMixin.has_config(self):
            QuantizationMixin.initialize_quantization(self, state.model)
        # 驗證duo_scaling只與per-channel量化策略兼容
        if self.duo_scaling:
            for _, module in match_named_modules(
                state.model, self.resolved_targets, self.ignore
            ):
                if hasattr(module, "quantization_scheme") and hasattr(module.quantization_scheme, "weights"):
                    if module.quantization_scheme.weights.strategy == QuantizationStrategy.TENSOR:
                        raise ValueError(
                            "duo_scaling is only supported with per-channel quantization "
                            "strategies (group or channel), but found TENSOR strategy. "
                            "Please set duo_scaling=False or use a per-channel "
                            "quantization strategy."
                        )

        # prepare module names
        self._module_names = {
            m: name
            for name, m in match_named_modules(
                state.model, self.resolved_targets, self.ignore
            )
        }

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # register quantization calibration hooks
        QuantizationMixin.start_calibration(self, state.model)

        # register gptq hooks
        added_hook = False
        for _, module in match_named_modules(
            state.model, self.resolved_targets, self.ignore
        ):
            if getattr_chain(module, "quantization_scheme.weights", None) is not None:
                if not isinstance(module, torch.nn.Embedding):
                    self.register_hook(module, self.calibrate_module, "forward")
                    added_hook = True

        if not added_hook:
            raise ValueError(
                "GPTQModifier requires a weight quantization config be specified by "
                "this modifier or a modifier preceding it"
            )

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self.compress_modules()

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            self.compress_modules()

            if not self.ended_:
                self.on_end(state, None)

    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        output: torch.Tensor,  # 改為使用output，不再是_unused
    ):
        """
        Calibration hook used to accumulate:
        1. Hessian of the input (for GPTQ)
        2. Output activation statistics (for AWQ duo-scaling)
        """
        # For GPTQ: accumulate Hessian from input
        inp = args[0]

        # Initialize hessian if not present
        if module not in self._num_samples:
            init_device = (
                "cpu" if self.offload_hessians else get_execution_device(module)
            )
            self._hessians[module] = make_empty_hessian(module, device=init_device)
            self._num_samples[module] = 0
        
        # For AWQ duo-scaling: accumulate output activation statistics
        if self.duo_scaling:
            if module not in self._activation_means:
                # Initialize activation statistics per output channel
                out_features = module.weight.shape[0]  # Linear層的輸出通道數
                self._activation_means[module] = torch.zeros(
                    out_features, device=output.device, dtype=torch.float64
                )
                self._activation_counts[module] = 0
            
            # Calculate absolute output activations
            # output shape: [batch_size, seq_len, out_features] or [batch_size, out_features]
            if output.dim() == 3:
                out_flat = output.detach().abs().flatten(0, -2)  # [batch*seq_len, out_features]
            else:
                out_flat = output.detach().abs()  # [batch_size, out_features]
            
            # Accumulate statistics
            self._activation_means[module] += out_flat.sum(dim=0).to(torch.float64)
            self._activation_counts[module] += out_flat.shape[0]

        # Accumulate hessian with input with optional offloading
        with self._maybe_onload_hessian(module):
            self._hessians[module], self._num_samples[module] = accumulate_hessian(
                inp,
                module,
                self._hessians[module],
                self._num_samples[module],
            )

    def compress_modules(self):
        """
        Quantize modules with optional AWQ duo-scaling
        """
        for module in list(self._num_samples.keys()):
            name = self._module_names[module]
            num_samples = self._num_samples[module]
            quant_args = getattr_chain(module, "quantization_scheme.weights")

            logger.info(f"Quantizing {name} using {num_samples} samples")
            logger.info(f"GPTQ for AWQ_N")
            # Apply AWQ duo-scaling if enabled
            scale_view = None
            
            if self.duo_scaling and module in self._activation_means:
                # Calculate duo-scaling factors (fixed ratio=0.5)
                scales = self._compute_duo_scales(module, ratio=0.5)
                scale_view = scales.view(-1, 1).to(module.weight.device)  # CORRECTED: shape [out_features, 1]
                
                # Apply scaling to weights: W * s (per output channel)
                module.weight.data.mul_(scale_view)
                
                logger.debug(f"Applied duo-scaling to {name}, scales shape: {scales.shape}")
            
            with (
                torch.no_grad(),
                align_module_device(module),
                self._maybe_onload_hessian(module),
                CompressionLogger(module) as comp_logger,
            ):
                # Quantize the scaled weights: Q(W * s)
                loss, quantized_weight, scale, zero_point, g_idx = quantize_weight(
                    module=module,
                    quant_args=quant_args,
                    hessians_dict=self._hessians,
                    blocksize=self.block_size,
                    percdamp=self.dampening_frac,
                )
                comp_logger.set_loss(loss)
            
            # If duo-scaling was applied, restore the weights
            if self.duo_scaling and scale_view is not None:
                # Quantized weight is Q(W * s), need to restore: Q(W * s) / s
                quantized_weight = quantized_weight / scale_view
            
            update_offload_parameter(module, "weight", quantized_weight)
            update_offload_parameter(module, "weight_scale", scale)
            update_offload_parameter(module, "weight_zero_point", zero_point)
            if g_idx is not None:
                update_offload_parameter(module, "weight_g_idx", g_idx)
            
            # Clean up activation statistics
            if self.duo_scaling:
                if module in self._activation_means:
                    del self._activation_means[module]
                if module in self._activation_counts:
                    del self._activation_counts[module]
            
            # self._hessians[module] already deleted by quantize_weight
            del self._num_samples[module]
    
    def _compute_weight_means(self, module: torch.nn.Module) -> torch.Tensor:
        """
        Compute per-output-channel mean of absolute weights (AWQ style)
        """
        if not hasattr(module, 'weight'):
            return torch.ones(1, device=module.weight.device)
        
        weight = module.weight  # [out_features, in_features]
        
        # AWQ style: directly compute mean of absolute values
        # No normalization or complex reshaping needed
        weight_mean = weight.abs().mean(dim=1)  # shape: [out_features]
        
        return weight_mean
    
    def _compute_duo_scales(self, module: torch.nn.Module, ratio: float = 0.5) -> torch.Tensor:
        """
        Compute duo-scaling factors using AWQ formula: s = (x^ratio) / (w^(1-ratio))
        
        Args:
            module: The module to compute scales for
            ratio: Balance between activation and weight (0.5 for equal balance)
        
        Returns:
            Scaling factors for each output channel
        """
        if module not in self._activation_means:
            # No activation statistics, return identity scaling
            return torch.ones(module.weight.shape[0], device=module.weight.device)
        
        # Compute activation mean (per output channel)
        act_mean = self._activation_means[module] / self._activation_counts[module]
        act_mean = act_mean.to(module.weight.device)
        
        # Compute weight mean (per output channel)
        weight_mean = self._compute_weight_means(module)
        
        # AWQ duo-scaling formula: s = (x^ratio) / (w^(1-ratio))
        # Fixed ratio=0.5 for balanced scaling
        scales = (act_mean.pow(ratio) / (weight_mean.pow(1 - ratio) + 1e-8)).clamp(min=1e-8)
        logger.debug(f"DEBUG: raw scales min/max = {scales.min():.6f}/{scales.max():.6f}")
        # Normalize for numerical stability
        scale_max = scales.max()
        scale_min = scales.min()
        if scale_max > 0 and scale_min > 0:
            scales = scales / (scale_max * scale_min).sqrt()
        
        # Handle outliers
        scales[torch.isinf(scales)] = 1
        scales[torch.isnan(scales)] = 1
        
        return scales

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by removing observers and calibration hooks
        """
        self.ended_ = True
        QuantizationMixin.end_calibration(self, state.model)
        self.remove_hooks()  # remove gptq hooks

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        Clean up quantization observers and cached data
        """
        if not self.ended_:
            self.on_end(state, None)

        if len(self._num_samples) > 0:
            raise ValueError(f"Failed to compress {len(self._num_samples)} modules")

        self._hessians = dict()
        self._num_samples = dict()
        
        # Clean up AWQ related data
        if self.duo_scaling:
            self._activation_means.clear()
            self._activation_counts.clear()

        return True

    @contextlib.contextmanager
    def _maybe_onload_hessian(self, module: torch.nn.Module):
        if self.offload_hessians:
            device = get_execution_device(module)
            self._hessians[module] = self._hessians[module].to(device=device)

        yield

        if self.offload_hessians:
            if module in self._hessians:  # may have been deleted in context
                self._hessians[module] = self._hessians[module].to(device="cpu")