from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .dthink_pixel_agent import DThinkPixelAgent


class DThinkPixelAgentGRPO(DThinkPixelAgent):
    """
    Thin wrapper over DThinkPixelAgent exposing a helper to reuse its tokenizer /
    vision preprocessing during GRPO rollouts.
    """

    def build_inputs_from_messages(self, messages: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], int]:
        """
        Leverage the parent helper to obtain tokenized prompt (with vision/pose
        inputs) for a list of chat `messages`. Returns (inputs, prompt_len).
        """
        return self._build_model_inputs(messages)
