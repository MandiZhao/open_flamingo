from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip

from .flamingo import Flamingo
from .flamingo_lm import FlamingoLMMixin
from .text_lm import TextLLM, TextLMMixin
from .utils import extend_instance
from peft import LoraModel, LoraConfig
from peft.tuners import lora  
PREFIXES = [lora.LoraModel.prefix] #, lokr.LoKrModel.prefix, loha.LoHaModel.prefix, oft.OFTModel.prefix]
Configs = [lora.LoraConfig] #, loha.LoHaConfig, lokr.LoKrConfig, adalora.AdaLoraConfig, oft.OFTConfig]
Layers = (lora.layer.LoraLayer) #, loha.layer.LoHaLayer, lokr.layer.LoKrLayer, adalora.layer.AdaLoraLayer, oft.OFTLayer)


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = False,
    cache_dir: Optional[str] = None,
    no_vis_encoder: bool = False,
    no_vision: bool = False,
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
        freeze_lm_embeddings (bool, optional): whether to freeze LM input embeddings when configuring Perceiver.
        cache_dir (str, optional): path to cache directory for downloading OpenClip/HF weights.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    # print("loading vision encoder")
    if no_vis_encoder or no_vision:
        vision_encoder = None
        image_processor = None
    else:
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            clip_vision_encoder_path,
            pretrained=clip_vision_encoder_pretrained,
            cache_dir=cache_dir,
        )
        # set the vision encoder to output the visual features
        vision_encoder.visual.output_tokens = True
    # print("done with vision encoder, now loading language encoder")
    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    # hacks for MPT-1B, which doesn't have a get_input_embeddings method
    if "mpt-1b-redpajama-200b" in lang_encoder_path:

        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings

        extend_instance(lang_encoder, EmbeddingFnMixin)

    # convert LM to FlamingoLM
    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))
    print("initialized Flamingo model")
    if no_vis_encoder or no_vision:
        vis_dim = 1024
    else:
        vis_dim = open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"]["width"]
    model = Flamingo(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=vis_dim,
        no_vision=no_vision,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        no_vis_encoder=no_vis_encoder,
        **flamingo_kwargs,
    )

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    if not no_vision:
        model.perceiver.requires_grad_(True)
    model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    if not freeze_lm_embeddings:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
        # TODO: investigate also training the output embeddings when untied

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, text_tokenizer

def create_textlm_and_transforms(
    lang_encoder_path: str,
    tokenizer_path: str,
    finetune_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = False,
    cache_dir: Optional[str] = None,
    use_lora: bool = False,
    lora_r = 16,
    lora_alpha = 32,
    **flamingo_kwargs,
): 
    image_processor = None
    # print("done with vision encoder, now loading language encoder")
    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    print(f"Before adding special tokens, len: {len(text_tokenizer)}")
    # add Flamingo special tokens to the tokenizer
    # text_tokenizer.add_special_tokens(
    #     {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    # )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    print(f"After adding special tokens, len: {len(text_tokenizer)}")

    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    # hacks for MPT-1B, which doesn't have a get_input_embeddings method
    if "mpt-1b-redpajama-200b" in lang_encoder_path:
        raise NotImplementedError("MPT-1B not supported yet")

    # convert LM to FlamingoLM
    extend_instance(lang_encoder, TextLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    if use_lora:
        assert finetune_every_n_layers > 100, "finetune_every_n_layers must be large for Lora"
        lang_encoder = LoraModel(lang_encoder, LoraConfig(task_type="SEQ_2_SEQ_LM", r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.01), "default")
    
    print(f"Before resize_token_embeddings, len: {len(text_tokenizer)}")
    lang_encoder.resize_token_embeddings(len(text_tokenizer))
    print(f"After resize_token_embeddings, len: {len(text_tokenizer)}")
    # print("Initializing Text-only model")
    model = TextLLM( 
        lang_encoder,
        eoc_token_id=text_tokenizer.encode(text_tokenizer.eos_token)[-1], # NOTE this is different from flamingo!
        # media_token_id=text_tokenizer.encode("<image>")[-1],
        finetune_every_n_layers=finetune_every_n_layers, 
        use_lora=use_lora,
    )
    if use_lora:
        # Freeze all parameters
        model.requires_grad_(False)
        lang_encoder._mark_only_adapters_as_trainable(lang_encoder) 
    else:
        # Freeze all parameters
        model.requires_grad_(False)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

        # Unfreeze finetuning layers, and LM input embeddings
        for layer in model.lang_encoder.unfrozen_layers:
            layer.requires_grad_(True)
    
    if not freeze_lm_embeddings:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
        # TODO: investigate also training the output embeddings when untied
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_in_billion = num_params / 1e9
    print(
        f"Text-only LLM initialized with {params_in_billion} Billion trainable parameters"
    )

    return model, image_processor, text_tokenizer
        
def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "code_llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
}
