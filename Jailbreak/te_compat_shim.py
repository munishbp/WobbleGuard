"""Patch transformer_engine for legacy vortex 1.0.6 checkpoints.

Imported explicitly by jailbreak_pipeline.py. It was a sitecustomize.py, which
Python auto-runs in EVERY interpreter: importing transformer_engine here made
its import-time "pip show" spawn a child that re-ran this file, recursing until
the host died on 2026-08-30. An explicit import runs it in one process only.

vortex 1.0.6 saves fp8 state without the "recipe" key that this newer
transformer_engine expects. Both directions of the state round trip need to
tolerate that. Nothing else about the old stack changes.
"""

try:
    from transformer_engine.pytorch.module import base as _te_base

    _Base = _te_base.TransformerEngineBaseModule
    _orig_set = _Base.set_extra_state
    _orig_get = _Base.get_extra_state

    def _ensure_recipe(module):
        """Give fp8_meta the default recipe when the checkpoint carries none."""
        meta = getattr(module, "fp8_meta", None)
        if isinstance(meta, dict) and "recipe" not in meta:
            from transformer_engine.common.recipe import DelayedScaling

            meta["recipe"] = DelayedScaling()

    def _tolerant_get_extra_state(self):
        # get_extra_state reads fp8_meta["recipe"] unconditionally.
        _ensure_recipe(self)
        return _orig_get(self)

    def _tolerant_set_extra_state(self, state):
        _ensure_recipe(self)
        try:
            return _orig_set(self, state)
        except KeyError as e:
            if "recipe" not in str(e):
                raise
            # Load what the old checkpoint does carry and leave the recipe at
            # the default set above.
            if isinstance(state, dict):
                for k, v in state.items():
                    if k != "recipe":
                        try:
                            self.fp8_meta[k] = v
                        except Exception:
                            pass
            return None

    _Base.get_extra_state = _tolerant_get_extra_state
    _Base.set_extra_state = _tolerant_set_extra_state
    print("[shim] transformer_engine get/set_extra_state patched for legacy checkpoints")
except Exception as _e:  # noqa: BLE001
    print(f"[shim] transformer_engine not patched: {_e}")
