import torch
from einops import rearrange
from torch import nn 
from .helpers import PerceiverResampler
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from .utils import apply_with_stopping_condition 
from .utils import getattr_recursive, setattr_recursive

""" Wraps around the base LM for text-only LLM baselines. No additional layers added."""

class TextLMMixin(nn.Module):
    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)
    
    def init_text_llm(
        self, 
        lang_hidden_size,
        finetune_every_n_layers,
    ):
        self.old_decoder_blocks = self._get_decoder_layers() # torch.nn.modules.container.ModuleList
        self.all_decoder_blocks = self._get_decoder_layers() 
        self.unfrozen_layers = []
        print("Original LM has {} decoder layers".format(len(self.old_decoder_blocks) ))  
        if finetune_every_n_layers > 0:
            unfrozen_layers = []
            old_layers = [] 
            for i, layer in enumerate(self.old_decoder_blocks):
                if i % finetune_every_n_layers == 0:
                    unfrozen_layers.append(layer) 
                else:
                    old_layers.append(layer)
            self.old_decoder_blocks = nn.ModuleList(old_layers)
            # old decoder blocks are frozen
            self.unfrozen_layers = nn.ModuleList(unfrozen_layers)
            if len(unfrozen_layers) == 0:
                print("Warning: No layers are unfrozen!")
        elif finetune_every_n_layers == 0: # finetune only the first layer
            self.old_decoder_blocks = self.old_decoder_blocks[1:]
            self.unfrozen_layers = self.old_decoder_blocks[:1] 
        elif finetune_every_n_layers <= -1: # only the last layer(s)
            self.old_decoder_blocks = self.old_decoder_blocks[:finetune_every_n_layers]
            self.unfrozen_layers = self.old_decoder_blocks[finetune_every_n_layers:] 
        else:
            raise NotImplementedError 
    
    def forward(self, input_ids, attention_mask,  **kwargs):
        # media_locations = input_ids == self.media_token_id
        # for layer in self._get_decoder_layers():
        #     layer.condition_media_locations(media_locations)
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        return super().forward(**kwargs)

    def is_conditioned(self):
        return all(l.is_conditioned() for l in self._get_decoder_layers())
    
    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_use_cached_media(None)

class TextLLM(nn.Module):
    def __init__(
        self,
        lang_encoder: nn.Module,
        eoc_token_id: int, 
        finetune_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
        use_lora: bool = False,
    ):
        super().__init__()
        self.lang_encoder = lang_encoder
        self.eoc_token_id = eoc_token_id 
        self.finetune_every_n_layers = finetune_every_n_layers
        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model
        else:
            self.lang_dim = lang_encoder.config.hidden_size
        self.lang_encoder.init_text_llm( 
            lang_hidden_size=self.lang_dim,
            finetune_every_n_layers=finetune_every_n_layers,
        )
        self.use_lora = use_lora
    
    def forward(
        self, 
        vision_x: torch.Tensor, # dummy
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        past_key_values=None,
        use_cache=False,
    ): 
        # self._condition_media_locations(input_ids=lang_x) 
        outputs = self.lang_encoder(
            lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return outputs          
        
    def generate(
        self,
        vision_x: torch.Tensor, # dummy
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        num_beams = kwargs.pop("num_beams", 1)
        eos_token_id = kwargs.pop("eos_token_id", self.eoc_token_id)
        output = self.lang_encoder.generate(
            input_ids=lang_x,
            attention_mask=attention_mask,
            num_beams=num_beams,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        return output   

    def wrap_fsdp(self, wrapper_kwargs, device_id):
        if self.use_lora:
            self.wrap_lora_fsdp(wrapper_kwargs, device_id)
        
        else:
            for block in self.lang_encoder.old_decoder_blocks:
                block.requires_grad_(False)
            for block in self.lang_encoder.unfrozen_layers:
                block.requires_grad_(True)

            with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
                self.lang_encoder.old_decoder_blocks = nn.ModuleList(
                    wrap(wrap(block)) for block in self.lang_encoder.old_decoder_blocks
                )
                self.lang_encoder.unfrozen_layers = nn.ModuleList(
                    wrap(wrap(block)) for block in self.lang_encoder.unfrozen_layers
                )
                # self.lang_encoder._set_decoder_layers(
                #     nn.ModuleList(wrap(wrap(block)) for block in self.lang_encoder._get_decoder_layers())
                # )

                self.lang_encoder.set_input_embeddings(
                    wrap(wrap(self.lang_encoder.get_input_embeddings()))
                )
                self.lang_encoder.set_output_embeddings(
                    wrap(wrap(self.lang_encoder.get_output_embeddings()))
                )

            # manually move non-FSDP managed parameters to device_id
            # these are all in lang_encoder
            apply_with_stopping_condition(
                module=self.lang_encoder,
                apply_fn=lambda m: m.to(device_id),
                apply_condition=lambda m: len(list(m.children())) == 0,
                stopping_condition=lambda m: isinstance(m, FSDP),
            )

            # exclude the original decoder layers from the optimizer
            for block in self.lang_encoder.old_decoder_blocks:
                for p in block.parameters():
                    p.exclude_from_optimizer = True

        # set up clip_grad_norm_ function
        def clip_grad_norm_(max_norm): 
            self.lang_encoder.get_input_embeddings().clip_grad_norm_(max_norm)

        self.clip_grad_norm_ = clip_grad_norm_
    
    def wrap_lora_fsdp(self, wrapper_kwargs, device_id):
        """ assume lora trainable params are already marked """ 
        from peft.tuners import lora
        LORA_PREFIX = lora.LoraModel.prefix
        # self.lang_encoder.requires_grad_(False)
        with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
            for layer in self.lang_encoder.model.model.layers:
                proj_layers = [layer.self_attn.q_proj, layer.self_attn.v_proj]
                for proj in proj_layers:
                    # proj.base_layer.requires_grad_(False)
                    proj.base_layer = wrap(wrap(proj.base_layer))
                    for p in proj.base_layer.parameters():
                        p.exclude_from_optimizer = True
                    
                    proj.lora_dropout = nn.ModuleDict(
                        {k: wrap(wrap(v)) for k, v in proj.lora_dropout.items()}
                    )
                    # for p in zip(proj.lora_A.parameters(), proj.lora_B.parameters()):
                    #     p.requires_grad = True 
                    
                    proj.lora_A = nn.ModuleDict(
                        {k: wrap(wrap(v)) for k, v in proj.lora_A.items()}
                    )
                    
                    proj.lora_B = nn.ModuleDict(
                        {k: wrap(wrap(v)) for k, v in proj.lora_B.items()}
                    ) 

                    proj.lora_embedding_A = wrap(wrap(proj.lora_embedding_A))
                    proj.lora_embedding_B = wrap(wrap(proj.lora_embedding_B))

                layer.self_attn.k_proj = wrap(wrap(layer.self_attn.k_proj))
                layer.self_attn.o_proj = wrap(wrap(layer.self_attn.o_proj))
                # layer.self_attn.rotary_emb = wrap(wrap(layer.self_attn.rotary_emb))
                for p in layer.self_attn.k_proj.parameters():
                    # p.require_grad_(False)
                    p.exclude_from_optimizer = True
                for p in layer.self_attn.o_proj.parameters():
                    # p.require_grad_(False)
                    p.exclude_from_optimizer = True
                
                for name in ['mlp', 'input_layernorm', 'post_attention_layernorm']:
                    setattr(layer, name, wrap(wrap(getattr(layer, name))))
                    for p in getattr(layer, name).parameters():
                        # p.require_grad_(False)
                        p.exclude_from_optimizer = True

            self.lang_encoder.set_input_embeddings(
                wrap(wrap(self.lang_encoder.get_input_embeddings()))
            )
            self.lang_encoder.set_output_embeddings(
                wrap(wrap(self.lang_encoder.get_output_embeddings()))
            )

        # manually move non-FSDP managed parameters to device_id
        # these are all in lang_encoder
        apply_with_stopping_condition(
            module=self.lang_encoder,
            apply_fn=lambda m: m.to(device_id),
            apply_condition=lambda m: len(list(m.children())) == 0,
            stopping_condition=lambda m: isinstance(m, FSDP),
        )
 