from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation


class Qwen2_5_VLVisionConfig(PretrainedConfig):
    model_type = "qwen2_5_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        hidden_size=3584,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=[7, 15, 23, 31],
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range


class Qwen2_5_VLTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2_5_VLTextModel`]. It is used to instantiate a
    Qwen2-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-VL-7B-Instruct [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 152064):
            Vocabulary size of the Qwen2_5_VL model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2_5_VLModel`]
        hidden_size (`int`, *optional*, defaults to 8192):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 29568):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 80):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 80):
            The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
            additional layer afterwards will use SWA (Sliding Window Attention).
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE

    ```python
    >>> from transformers import Qwen2_5_VLTextModel, Qwen2_5_VLConfig

    >>> # Initializing a Qwen2_5_VL style configuration
    >>> configuration = Qwen2_5_VLConfig()

    >>> # Initializing a model from the Qwen2-VL-7B style configuration
    >>> model = Qwen2_5_VLTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_5_vl_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `Qwen2_5_VL`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=8192,
        intermediate_size=29568,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=80,
        layer_types=None,
        attention_dropout=0.0,
        rope_scaling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        # and change type from 'mrope' to 'default' because `mrope` does default RoPE calculations
        # one can set it to "linear"/"dynamic" etc. to have scaled RoPE
        # TODO: @raushan update config in the hub
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class DThink_ActionConfig(PretrainedConfig):
    """Configuration for the optional action head controlled by a special token.

    This config enables a control token (Qwen-style) that, when present in the input,
    taps a specific transformer layer's hidden state at that token position and pushes
    it through an MLP head to predict a fixed number of (x, y, sin(theta), cos(theta)) quadruplets.

    Notes:
    - The modeling code is expected to look up this config and, when `use_action_head`
      is True and the `action_token`/`action_token_id` is found at some positions,
      route the hidden state(s) through the head. This class only declares knobs; it
      does not implement the head itself.
    - We keep the token string in Qwen special-token style: "<|action|>".
    - The MLP defaults (hidden size, intermediate size, activation) fall back to the
      text config unless overridden here.
    """

    model_type = "qwen2_5_vl"
    base_config_key = "action_config"

    def __init__(
        self,
        hidden_size=3584,
        hidden_act="silu",
        intermediate_size=3420,
        bias=True,
        layer_index: int = -1,
        triplet_count: int = 8,
        coordinate_mode: str = "relative",
        action_dropout: float = 0.0,
        theta_unit: str = "radian",
        translation_scale: float = 1.0,
        theta_scale: float = 1.0,
        translation_bias: float = 0.0,
        theta_bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.hidden_act = hidden_act
        self.bias = bias

        self.layer_index = int(layer_index)
        self.triplet_count = int(triplet_count)

        if coordinate_mode not in ("absolute", "relative"):
            raise ValueError("coordinate_mode must be 'absolute' or 'relative', got %r" % coordinate_mode)
        self.coordinate_mode = coordinate_mode

        self.action_dropout = float(action_dropout)

        if theta_unit not in ("radian", "degree"):
            raise ValueError("theta_unit must be 'radian' or 'degree', got %r" % theta_unit)
        self.theta_unit = theta_unit

        self.translation_scale = float(translation_scale)
        self.theta_scale = float(theta_scale)
        
        self.translation_bias = float(translation_bias)
        self.theta_bias = float(theta_bias)

    @property
    def output_size(self) -> int:
        return int(self.triplet_count) * 4


class Qwen2_5_VLConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2_5_VLModel`]. It is used to instantiate a
    Qwen2-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-VL-7B-Instruct [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen2_5_VLTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Qwen2_5_VLVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.
        action_nav_token_id (`int`, *optional*, defaults to 151665):
            The token index to denote navigation action input.
        action_pad_token_id (`int`, *optional*, defaults to 151671):
            The token index to denote navigation action input.
        pos_token_id (`int`, *optional*, defaults to 151670):
            The token index to denote pose embeddings input.
        pos_rope_dim (`int`, *optional*, defaults to `text_config.hidden_size`):
            The feature dimension used inside the RoPE-based pose encoder. Must be divisible by 4.
        pos_hidden_dim (`int`, *optional*, defaults to `text_config.hidden_size`):
            The hidden dimension for the MLP in the pose encoder.
        pos_resolution (`float`, *optional*, defaults to 0.1):
            Spatial resolution used to bucket (x, y) values before applying RoPE.
        theta_resolution (`float`, *optional*, defaults to 0.01):
            Spatial resolution used to bucket (x, y) values before applying RoPE.
        pos_base (`float`, *optional*, defaults to 10000.0):
            Base value for rotary embeddings in the pose encoder.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The token index to denote start of vision input.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The token index to denote end of vision input.

    ```python
    >>> from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig

    >>> # Initializing a Qwen2_5_VL style configuration
    >>> configuration = Qwen2_5_VLConfig()

    >>> # Initializing a model from the Qwen2-VL-7B style configuration
    >>> model = Qwen2_5_VLForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_5_vl"
    sub_configs = {
        "vision_config": Qwen2_5_VLVisionConfig,
        "text_config": Qwen2_5_VLTextConfig,
        "action_config": DThink_ActionConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        action_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        action_nav_token_id=151665,
        action_pad_token_id=151671,
        pos_token_id=151670,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        **kwargs,
    ):
        # We need to init super() here so that it does not reset values
        # that are in text config to the BaseClass defaults. The Base
        # config has many text related defaults and not all defaults are same as for `Qwen2_5_VLTextConfig`
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
            
        if isinstance(action_config, dict):
            self.action_config = self.sub_configs["action_config"](**action_config)
        elif action_config is None:
            self.action_config = self.sub_configs["action_config"](**kwargs)

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.action_nav_token_id = action_nav_token_id
        self.action_pad_token_id = action_pad_token_id
        self.pos_token_id = pos_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        # Attention implementation to use. It sets it recursively on sub-configs so we call it again in the end
        self._attn_implementation = kwargs.pop("attn_implementation", None)
        self.pos_rope_dim = kwargs.pop("pos_rope_dim", None)
        self.pos_hidden_dim = kwargs.pop("pos_hidden_dim", None)
        self.pos_resolution = kwargs.pop("pos_resolution", 0.1)
        self.theta_resolution = kwargs.pop("theta_resolution", 0.01)
        self.pos_base = kwargs.pop("pos_base", 10000.0)

    def __setattr__(self, key, value):
        if (
            (text_config := super().__getattribute__("__dict__").get("text_config")) is not None
            and key not in ["dtype", "_attn_implementation_internal"]
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        if "text_config" in super().__getattribute__("__dict__") and key not in [
            "dtype",
            "_attn_implementation_internal",
        ]:
            text_config = super().__getattribute__("text_config")
            if key in text_config.__dict__:
                return getattr(text_config, key)

        return super().__getattribute__(key)


__all__ = [
    "Qwen2_5_VLConfig",
    "Qwen2_5_VLTextConfig",
    "Qwen2_5_VLVisionConfig",
    "DThink_ActionConfig",
]
