"""Phase 7 Step 8: describes how the AVAILABLE independent models relate to each other -
never a claim about profitability (Step 8: "Do not assume agreement automatically means
profitable")."""

from __future__ import annotations

from nfl_predict.decision.schemas import ModelAgreementDescriptor, ModelPoint


def compute_model_agreement(elo: ModelPoint, ridge: ModelPoint, lightgbm: ModelPoint) -> ModelAgreementDescriptor:
    points = [(m.model_id, m.predicted_margin) for m in (elo, ridge, lightgbm) if m.available and m.predicted_margin is not None]
    models_considered = tuple(p[0] for p in points)
    n = len(points)

    if n < 2:
        return ModelAgreementDescriptor(models_considered=models_considered, all_agree_on_direction=None, margin_dispersion=None, n_models_available=n)

    margins = [p[1] for p in points]
    signs = {(-1 if m < 0 else (1 if m > 0 else 0)) for m in margins}
    all_agree = len(signs) == 1
    dispersion = max(margins) - min(margins)
    return ModelAgreementDescriptor(models_considered=models_considered, all_agree_on_direction=all_agree, margin_dispersion=dispersion, n_models_available=n)
