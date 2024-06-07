from typing import List, Optional, Tuple, Union
from camml.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from camml.model.bert import BertConfig, BertLMHeadModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import random

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoTokenizer, T5EncoderModel
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class CaMMLConfig(LlamaConfig):
    model_type = "CaMML"


class CaMML(LlavaMetaModel, LlamaModel):
    config_class = CaMMLConfig

    def __init__(self, config: LlamaConfig):
        super(CaMML, self).__init__(config)


class CaMML_LM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = CaMMLConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = CaMML(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        num_query_token = config.perceiver_querys # 64
        perceiver_input_width = 768
        self.perceiver_layers_single_modality = getattr(config, "perceiver_layers", 2)
        self.perceiver_layers = getattr(config, "perceiver_layers", 2)
        self.perceiver_from_pretrained = getattr(config, "perceiver_from_pretrained", True)
        self.random_shots_training = getattr(config, "random_shots_training", False)
        self.random_shots_max = getattr(config, "random_shots_max", 3)
        self.perceiver_hiddensize = getattr(config, "perceiver_hidden_size", 768)
        self.config = config
        self.context_perceiver, self.query_tokens_vision, self.query_tokens_text = self.init_Perceiver(
            num_query_token, perceiver_input_width)
        self.context_perceiver.bert.embeddings.word_embeddings = None
        self.context_perceiver.bert.embeddings.position_embeddings = None
        for layer in self.context_perceiver.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.vision_perceiver = self.init_Perceiver_single_modality()
        self.language_perceiver = self.init_Perceiver_single_modality()

        self.vision_perceiver.bert.embeddings.word_embeddings = None
        self.vision_perceiver.bert.embeddings.position_embeddings = None
        for layer in self.vision_perceiver.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.language_perceiver.bert.embeddings.word_embeddings = None
        self.language_perceiver.bert.embeddings.position_embeddings = None
        for layer in self.language_perceiver.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.context_proj_vision = nn.Linear(self.perceiver_hiddensize, config.hidden_size)
        self.context_proj_text = nn.Linear(self.perceiver_hiddensize, config.hidden_size)

        self.image_proj_to_perceiver = nn.Linear(config.hidden_size, self.perceiver_hiddensize)
        self.text_proj_to_perceiver = nn.Linear(config.hidden_size, self.perceiver_hiddensize)

    def init_Perceiver_single_modality(self, max_position_embeddings=4096, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = self.perceiver_hiddensize
        encoder_config.hidden_size = self.perceiver_hiddensize
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = 256
        encoder_config.max_position_embeddings = max_position_embeddings
        encoder_config.num_hidden_layers = self.perceiver_layers_single_modality
        encoder_config.gradient_checkpointing = False
        if self.perceiver_from_pretrained:
            perceiver = BertLMHeadModel.from_pretrained(
                "bert-base-uncased", config=encoder_config, ignore_mismatched_sizes=True
            )
        else:
            perceiver = BertLMHeadModel(config=encoder_config)

        return perceiver

    def init_Perceiver(self, num_query_token, vision_width, max_position_embeddings=4096, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = self.perceiver_hiddensize
        encoder_config.hidden_size = self.perceiver_hiddensize
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.max_position_embeddings = max_position_embeddings
        encoder_config.num_hidden_layers = self.perceiver_layers
        encoder_config.gradient_checkpointing = False
        if self.perceiver_from_pretrained:
            perceiver = BertLMHeadModel.from_pretrained(
                "bert-base-uncased", config=encoder_config, ignore_mismatched_sizes=True
            )
        else:
            perceiver = BertLMHeadModel(config=encoder_config)

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        query_tokens_2 = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens_2.data.normal_(mean=0.0, std=encoder_config.initializer_range)


        return perceiver, query_tokens, query_tokens_2

    def get_model(self):
        return self.model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_ret: Optional[torch.FloatTensor] = None,
        context: Optional[List[List[str]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.random_shots_training:
            random_shots = random.randint(1, self.random_shots_max)
            images_ret = [images_ret_bs[:random_shots] if len(images_ret_bs[:random_shots].shape) == 4
                          else images_ret_bs[:random_shots].unsqueeze(0) for images_ret_bs in images_ret]
            context = [context_bs[:random_shots] for context_bs in context]


        input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images, images_ret, context)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, images_ret, context
    ):
        vision_tower = self.get_vision_tower()
        mm_projector = self.get_model().mm_projector
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            images_ret_stack = torch.cat(images_ret)
            image_ret_features = vision_tower(images_ret_stack)
            bs = images.shape[0]
            ret_num = images_ret[0].shape[0]
            image_features = self.encode_images(images)
            image_ret_features = self.get_model().mm_projector(image_ret_features)
            image_ret_features = torch.split(image_ret_features, [ret_num] * bs)


        image_ret_features_bs_ret_list = []  # [ [bs, L, C], [ret2], [ret3] ]
        context_bs_ret_list = []
        for ret_i in range(ret_num):
            image_ret_num_i = []
            for image_feat in image_ret_features:
                image_feat_1LC = image_feat[ret_i].unsqueeze(0)
                image_ret_num_i.append(image_feat_1LC)

            context_num_i = []
            for context_bs in context:
                context_num_i.append(context_bs[ret_i])

            context_bs_ret_list.append(context_num_i)
            image_ret_features_bs_ret_list.append(torch.cat(image_ret_num_i))

        context_querys_vision = []
        context_querys_text = []
        for context_ret_i, image_feat_ret_i in zip(context_bs_ret_list, image_ret_features_bs_ret_list):

            context_ret_i_ids = self.camml_tokenizer(
                context_ret_i,
                return_tensors="pt",
                padding="longest",
                max_length=2048,
                truncation=True,
            ).to("cuda")

            context_text_embeds = self.get_model().embed_tokens(context_ret_i_ids.input_ids)
            context_text_embeds = self.text_proj_to_perceiver(context_text_embeds)

            context_image_feat = self.image_proj_to_perceiver(image_feat_ret_i)  # [bs, 256, 768]
            context_atts = torch.ones(context_text_embeds.size()[:-1], dtype=torch.long).to(
                context_text_embeds.device)

            query_output = self.vision_perceiver.bert(
                query_embeds=context_image_feat, # [bs, 256, 768]
                encoder_hidden_states=context_text_embeds, # [bs, length, 768]
                encoder_attention_mask=context_atts,
                return_dict=True,
            )  # [bs, 256, 768]
            query_output_vision = query_output.last_hidden_state[:, :context_image_feat.size(1), :]

            context_atts = torch.ones(context_image_feat.size()[:-1], dtype=torch.long).to(
                context_image_feat.device)
            query_output = self.language_perceiver.bert(
                query_embeds=context_text_embeds,
                encoder_hidden_states=context_image_feat,
                encoder_attention_mask=context_atts,
                return_dict=True,
            )  # [bs, 256, 768]
            query_output_text = query_output.last_hidden_state[:, :context_text_embeds.size(1), :]

            context_querys_vision.append(query_output_vision)
            context_querys_text.append(query_output_text)

        context_querys_vision = torch.cat(context_querys_vision, dim=1)  # [bs, 256*ret, 768]
        context_querys_text = torch.cat(context_querys_text, dim=1)  # [bs, length * ret, 768]

        context_atts = torch.ones(context_querys_vision.size()[:-1], dtype=torch.long).to(
            context_querys_vision.device)
        query_tokens_vision = self.query_tokens_vision.expand(context_querys_vision.shape[0], -1, -1)
        query_output_vision = self.context_perceiver.bert(
            query_embeds=query_tokens_vision,
            encoder_hidden_states=context_querys_vision,
            encoder_attention_mask=context_atts,
            return_dict=True,
        )
        context_inputs_llm_vision = self.context_proj_vision(
            query_output_vision.last_hidden_state[:, :query_tokens_vision.size(1), :])

        context_atts = torch.ones(context_querys_text.size()[:-1], dtype=torch.long).to(
            context_querys_vision.device)
        query_tokens_text = self.query_tokens_text.expand(context_querys_text.shape[0], -1, -1)
        query_output_text = self.context_perceiver.bert(
            query_embeds=query_tokens_text,
            encoder_hidden_states=context_querys_text,
            encoder_attention_mask=context_atts,
            return_dict=True,
        )
        context_inputs_llm_text = self.context_proj_text(
            query_output_text.last_hidden_state[:, :query_tokens_text.size(1), :])

        context_inputs_llm = torch.cat([context_inputs_llm_vision, context_inputs_llm_text], dim=1)  # 512

        new_input_embeds = []
        new_labels = [] if labels is not None else None

        for batch_idx, cur_input_ids in enumerate(input_ids):

            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))

            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 1:
                # multimodal LLM, but the current sample is not multimodal
                # half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[batch_idx]
                cur_ret_features = context_inputs_llm[batch_idx]

                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                image_token_start = image_token_indices[0]

                if labels is not None:
                    cur_labels = labels[batch_idx]

                    cur_new_labels = []
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(
                        torch.full((cur_ret_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                   dtype=labels.dtype))
                    cur_new_labels.append(cur_labels[image_token_start + 1:])

                    new_labels.append(torch.cat(cur_new_labels))

                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                after_ret_pure_text_ids = cur_input_ids[image_token_start+1:]
                half_len = after_ret_pure_text_ids.shape[0] // 2

                cur_input_embeds_2 = self.get_model().embed_tokens(after_ret_pure_text_ids[:half_len])
                cur_input_embeds_3 = self.get_model().embed_tokens(after_ret_pure_text_ids[half_len:])

                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_ret_features, cur_input_embeds_2, cur_image_features[0:0], cur_input_embeds_3], dim=0)

                new_input_embeds.append(cur_input_embeds)
                continue

            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 2:
                cur_image_features = image_features[batch_idx]
                cur_ret_features = context_inputs_llm[batch_idx]

                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                ret_token_start = image_token_indices[0]
                image_token_start = image_token_indices[1]

                if labels is not None:
                    cur_labels = labels[batch_idx]

                    cur_new_labels = []
                    cur_new_labels.append(cur_labels[:ret_token_start])
                    cur_new_labels.append(
                        torch.full((cur_ret_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                   dtype=labels.dtype))

                    cur_new_labels.append(cur_labels[ret_token_start + 1: image_token_start])
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                   dtype=labels.dtype))
                    cur_new_labels.append(cur_labels[image_token_start + 1:])

                    new_labels.append(torch.cat(cur_new_labels))

                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:ret_token_start])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[ret_token_start + 1: image_token_start])

                cur_input_embeds_3 = self.get_model().embed_tokens(cur_input_ids[image_token_start+1: ])

                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_ret_features, cur_input_embeds_2, cur_image_features, cur_input_embeds_3], dim=0)
                new_input_embeds.append(cur_input_embeds)

                continue

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "images_ret": kwargs.get("images_ret", None),
                "context": kwargs.get("context", None),
            }
        )
        return model_inputs

AutoConfig.register("CaMML", CaMMLConfig)
AutoModelForCausalLM.register(CaMMLConfig, CaMML_LM)
