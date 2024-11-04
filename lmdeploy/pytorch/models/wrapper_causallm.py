import torch.nn as nn
import torch
from typing import Any, List, Optional
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager

class WrapperForCausalLM(nn.Module):
    def __init__(self, model):
        self.model = model
        self.ctx_mgr = StepContextManager()
        self.lm_head = model.lm_head
        
    def forward(
        self,
        # input_ids: torch.Tensor,
        # position_ids: torch.Tensor,
        # past_key_values: tuple[tuple[torch.Tensor]],
        # attention_mask: torch.Tensor,
        # inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """model forward, return logits."""
        first_token = kwargs.pop("first_token", False)
        if first_token and hasattr(self.model, "trace_graph_first"):
            outputs = self.model.trace_graph_first(**kwargs)
        else:
            outputs = self.model.trace_graph(**kwargs)
        logits, past_key_values, hidden_states = outputs
        # todo: kv cache update
        context = self.ctx_mgr.current_context()
        context._outputs["hidden_states"] = hidden_states
        context._outputs["logits"] = logits
        context._outputs["past_key_values"] = past_key_values
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        context = self.ctx_mgr.current_context()
        def _check_hidden_states(x, y):
            return torch.sum(torch.abs(x-y))<1e-9
        ctx_hidden_states = context._outputs.get("hidden_states", None)
        ctx_logits = context._outputs.get("logits", None)
        if ctx_hidden_states and _check_hidden_states(ctx_hidden_states, hidden_states) and ctx_logits:
            return ctx_logits
        return self.lm_head(hidden_states)

    def get_warmup_past_key_values(self, input_ids):
        input_bs = input_ids.size()[0]
        if hasattr(self.config, "n_layer"):
            num_hidden_layers = self.config.n_layer
        elif hasattr(self.config, "num_hidden_layers"):
            num_hidden_layers = self.config.num_hidden_layers
        elif hasattr(self.config, "num_layers"):
            num_hidden_layers = self.config.num_layers
        elif hasattr(self.config, "n_layers"):
            num_hidden_layers = self.config.n_layers
        beam_idx_tmp = torch.zeros(
            (2048, int(input_bs)), dtype=torch.long
        ).contiguous()
        return tuple(
            [
                (
                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    beam_idx_tmp,
                )
                for i in range(num_hidden_layers)
            ]
        )

    def prepare_inputs_for_generation(
        self,
        past_key_values: Optional[List[List[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attention_mask = context.attention_mask

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)
        
        first_token = past_key_values is None
        if past_key_values is None:
            past_key_values = self.get_warmup_past_key_values(input_ids)
        # inputs of forward
        model_inputs = dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        model_inputs = self.model.prepare_inputs_for_generation(**model_inputs)
        model_inputs["first_token"] = first_token
        return model_inputs