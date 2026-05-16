"""Ансамбль convnext_tiny + densenet121 + efficientnet_b1 (per-class soft vote)."""

from models.ensemble_i.ensemble import combine_member_probs

__all__ = ["combine_member_probs"]
