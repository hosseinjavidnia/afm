from __future__ import annotations

from .convnet_adapters import AFMConvNet, ScalarTransferConflictNet


def build_model(config: dict) -> AFMConvNet:
    model = config["model"]
    kind = str(model.get("kind", "convnet_adapters"))
    if kind == "scalar_transfer_conflict":
        if int(model.get("num_classes", 2)) != 2:
            raise ValueError("scalar_transfer_conflict requires exactly two classes")
        return ScalarTransferConflictNet()
    if kind != "convnet_adapters":
        raise ValueError(f"Unknown model.kind: {kind}")
    return AFMConvNet(
        num_classes=int(model["num_classes"]),
        feature_dim=int(model["feature_dim"]),
        adapter_bottleneck=int(model["adapter_bottleneck"]),
        adapter_slots=int(model["adapter_slots"]),
        initially_active_adapters=int(model["initially_active_adapters"]),
    )
