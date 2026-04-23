"""Register model presets. Add new models by defining a class and listing it here."""

from __future__ import annotations

from llm_training.player_summary.presets.gpt2 import GPT2Preset
from llm_training.player_summary.presets.mistral import MistralPreset
from llm_training.player_summary.presets.phi3_mini import Phi3MiniPreset
from llm_training.player_summary.presets.qwen35_27b_dflash import Qwen3_1_7BPreset

_PRESETS: dict[str, object] = {
    GPT2Preset.id: GPT2Preset(),
    MistralPreset.id: MistralPreset(),
    Phi3MiniPreset.id: Phi3MiniPreset(),
    Qwen3_1_7BPreset.id: Qwen3_1_7BPreset(),
}

PRESET_IDS: tuple[str, ...] = tuple(sorted(_PRESETS.keys()))


def get_preset(preset_id: str):
    if preset_id not in _PRESETS:
        raise KeyError(
            f"Unknown model preset {preset_id!r}. Known: {', '.join(PRESET_IDS)}"
        )
    return _PRESETS[preset_id]


def default_model_for_preset(preset_id: str) -> str:
    return get_preset(preset_id).default_model_id
