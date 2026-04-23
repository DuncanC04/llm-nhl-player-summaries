"""Map desired LoRA targets to layers that exist on a loaded model."""


def resolve_lora_targets(model, candidates):
    """Intersect candidate suffixes with module leaf names present on the model."""
    leaves = set()
    for name, _ in model.named_modules():
        if not name:
            continue
        leaves.add(name.rsplit(".", 1)[-1])
    resolved = [c for c in candidates if c in leaves]
    if not resolved:
        sample = sorted(leaves)
        raise ValueError(
            "No LoRA target modules matched this model. "
            f"Tried {candidates!r}. Example leaf names: {sample[:50]}"
        )
    return resolved
