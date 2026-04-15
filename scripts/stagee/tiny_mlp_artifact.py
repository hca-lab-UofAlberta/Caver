from __future__ import annotations

import math
from typing import Any
from typing import Sequence


def stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        denominator = 1.0 + math.exp(-value)
        return 1.0 / denominator
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def gelu(value: float) -> float:
    return 0.5 * value * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (value + (0.044715 * (value ** 3)))))


def bounded_sigmoid(value: float, *, lower: float, upper: float) -> float:
    if upper < lower:
        raise ValueError(f"invalid bounded sigmoid interval: {lower}..{upper}")
    return float(lower) + ((float(upper) - float(lower)) * stable_sigmoid(value))


def _apply_activation(values: list[float], activation: str | None) -> list[float]:
    if activation in (None, "", "identity"):
        return values
    if activation == "gelu":
        return [gelu(value) for value in values]
    if activation == "tanh":
        return [math.tanh(value) for value in values]
    raise ValueError(f"unsupported activation: {activation}")


def forward_linear_layer(inputs: Sequence[float], layer: dict[str, Any]) -> list[float]:
    outputs: list[float] = []
    for row, bias in zip(layer["weight"], layer["bias"]):
        total = float(bias)
        for weight, input_value in zip(row, inputs):
            total += float(weight) * float(input_value)
        outputs.append(total)
    return _apply_activation(outputs, str(layer.get("activation") or "identity"))


def forward_stack(inputs: Sequence[float], layers: Sequence[dict[str, Any]]) -> list[float]:
    values = [float(value) for value in inputs]
    for layer in layers:
        values = forward_linear_layer(values, layer)
    return values


def mean_and_population_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("mean/std requires at least one value")
    mean = float(sum(values) / float(len(values)))
    variance = sum((float(value) - mean) ** 2 for value in values) / float(len(values))
    return mean, math.sqrt(max(0.0, variance))


def serialize_torch_linear_layer(linear: Any, *, activation: str = "identity") -> dict[str, Any]:
    return {
        "weight": linear.weight.detach().cpu().tolist(),
        "bias": linear.bias.detach().cpu().tolist(),
        "activation": activation,
    }
