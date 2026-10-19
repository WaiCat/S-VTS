
import os
import json

import torch
import torch.nn as nn

from transformers import PreTrainedModel

from .predictor.topic_seg_predictor import TopicSegPredictor
from .vis_encoder.vis_encoder import VisEncoder
from .text_encoder.text_encoder import TextEncoder
from .projector.get_projector import get_projector
from .cross_encoder.get_cross_encoder import get_cross_encoder


torch.set_printoptions(threshold=float('inf'))
torch.set_printoptions(precision=4)


class MultiModalForTS(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.current_mode = "train"

        self.text_encoder = TextEncoder(config)
        self.vis_encoder = VisEncoder(config)

        self.projector = get_projector(config)
        self.cross_encoder = get_cross_encoder(config)

        self.predictor = TopicSegPredictor(config)
        self.use_silence = config.use_silence
        self.silence_dim = config.silence_dim

        if self.config.freeze_text_encoder:
            self.freeze_text_encoder()
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def set_mode(self, mode):
        assert mode in ["train", "dev", "test"]
        self.current_mode = mode
        print("self.current_mode: ", self.current_mode)
    
    def freeze_text_encoder(self):
        print("freeze_text_encoder is True.")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
    def select_class_values_with_batch(self, labels, text_features=None, vis_features=None, audio_features=None, silence_features=None, pause_features=None):
        ### 保留batch，需要记录有效特征的mask
        if silence_features is None:
            silence_features = pause_features

        # 创建一个mask来记录labels中0和1的位置 (即有效的标签)
        valid_labels_mask = (labels != -100)
        # 统计每个样本中有效标签的数量，并确定最大的有效序列长度
        valid_counts = valid_labels_mask.sum(dim=1)
        max_sequence_length = valid_counts.max().item()

        # 初始化padded文本特征和视觉特征，并初始化否定的mask以标记无效的向量位置
        padded_text_features, padded_vis_features, padded_audio_features, padded_silence_features = None, None, None, None
        if text_features is not None:
            batch_size, text_sequence, text_hidden_size = text_features.shape
            padded_text_features = torch.zeros((batch_size, max_sequence_length, text_hidden_size)).to(text_features.device)
            mask = torch.zeros((batch_size, max_sequence_length), dtype=torch.bool).to(text_features.device)

        if vis_features is not None:
            batch_size, vis_sequence, vis_hidden_size = vis_features.shape
            padded_vis_features = torch.zeros((batch_size, max_sequence_length, vis_hidden_size)).to(vis_features.device)
            mask = torch.zeros((batch_size, max_sequence_length), dtype=torch.bool).to(vis_features.device)

        if audio_features is not None:
            batch_size, audio_sequence, audio_hidden_size = audio_features.shape
            padded_audio_features = torch.zeros((batch_size, max_sequence_length, audio_hidden_size)).to(audio_features.device)
            mask = torch.zeros((batch_size, max_sequence_length), dtype=torch.bool).to(audio_features.device)

        if silence_features is not None:
            batch_size, silence_sequence, silence_hidden_size = silence_features.shape
            padded_silence_features = torch.zeros((batch_size, max_sequence_length, silence_hidden_size)).to(silence_features.device)
            mask = torch.zeros((batch_size, max_sequence_length), dtype=torch.bool).to(silence_features.device)

        for i in range(batch_size):
            # 获取当前样本的有效标签数量
            valid_count = valid_counts[i]

            # 提取有效特征,填充到最大序列长度
            if text_features is not None:
                valid_text = text_features[i, valid_labels_mask[i]]
                padded_text_features[i, :valid_count] = valid_text
            
            if vis_features is not None:
                valid_visual = vis_features[i, :valid_count]
                padded_vis_features[i, :valid_count] = valid_visual
            
            if audio_features is not None:
                valid_audio = audio_features[i, :valid_count]
                padded_audio_features[i, :valid_count] = valid_audio

            if silence_features is not None:
                valid_silence = silence_features[i, :valid_count]
                padded_silence_features[i, :valid_count] = valid_silence

            # 更新mask，标记有效的特征位置
            mask[i, :valid_count] = 1

        return padded_text_features, padded_vis_features, padded_audio_features, padded_silence_features, mask

    def restore_logits(self, labels, selected_logits):
        # 和去batch化的结果搭配使用
        # restore logits to (batch_size, sequence_length, num_labels), padding with zero
        batch_size, sequence_length = labels.shape
        mask = labels != -100
        valid_features_count = mask.sum(1)
        
        if self.config.out_modal_prob:
            restored_logits = torch.zeros(batch_size, sequence_length, self.config.num_labels * 3).to(selected_logits.device)
        else:
            restored_logits = torch.zeros(batch_size, sequence_length, self.config.num_labels).to(selected_logits.device)
        pointer = 0
        for i in range(batch_size):
            length = valid_features_count[i]
            restored_logits[i, :length] = selected_logits[pointer:pointer+length]
            pointer += length
        
        return restored_logits
    
    def extract_valid_vectors(self, batch_mask, batch_text_features=None, batch_vis_features=None, batch_audio_features=None, batch_silence_features=None, batch_pause_features=None):
        # 使用掩码选择有效的特征
        if batch_silence_features is None:
            batch_silence_features = batch_pause_features
        valid_text_vectors, valid_visual_vectors, valid_audio_vectors, valid_silence_vectors = None, None, None, None

        if batch_text_features is not None:
            valid_text_vectors = batch_text_features[batch_mask]

        if batch_vis_features is not None:
            valid_visual_vectors = batch_vis_features[batch_mask]
        
        if batch_audio_features is not None:
            valid_audio_vectors = batch_audio_features[batch_mask]

        if batch_silence_features is not None:
            valid_silence_vectors = batch_silence_features[batch_mask]
            
        return valid_text_vectors, valid_visual_vectors, valid_audio_vectors, valid_silence_vectors

    def apply_silence_encoder(self, text_features, silence_features):
        if (not self.use_silence) or silence_features is None or text_features is None:
            return text_features
        if self.silence_dim <= 0:
            return text_features
        silence_features = silence_features.to(text_features.device)
        # EXP-A (zero_silence_gate): keep every parameter and the full computation
        # graph, but drive the silence pathway with s_i = 0.
        if self.zero_silence:
            silence_features = torch.zeros_like(silence_features)
        gate_logits = self.silence_encoder_gate(silence_features) + self.silence_encoder_hidden_gate(text_features)
        gate = torch.sigmoid(gate_logits)
        # Same multiplicative gating form as the predictor stage, so the
        # injection-position ablation varies only the position, not the mechanism.
        return text_features * gate

    def apply_pause_encoder(self, text_features, pause_features):
        return self.apply_silence_encoder(text_features, pause_features)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
        example_ids=None,
        vis_ids=None,
        vis2d_embeds=None,
        vis3d_embeds=None,
        vis_ocr_embeds=None,
        silence_feat=None,
    ):
        device = input_ids.device
        loss = torch.tensor(0, dtype=torch.float, requires_grad=True).to(device)
            
        text_features, vis_features = None, None

        if self.config.fuse_type != "vis_only":
            text_features = self.text_encoder(
                input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask if head_mask is not None else None,
                token_type_ids=token_type_ids,
                position_ids=position_ids if position_ids is not None else None,
                inputs_embeds=inputs_embeds if inputs_embeds is not None else None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        if self.config.fuse_type != "text_only":
            vis_features = self.vis_encoder(self.current_mode, example_ids, vis_ids, vis2d_embeds, vis3d_embeds, vis_ocr_embeds)
        
        batch_text_features, batch_vis_features, batch_audio_features, batch_silence_features, batch_mask = self.select_class_values_with_batch(
            labels,
            text_features,
            vis_features,
            None,
            silence_features=silence_feat,
        )
        batch_text_features, batch_vis_features, _ = self.projector(batch_text_features, batch_vis_features, None)

        valid_projected_text_vectors, valid_projected_visual_vectors, valid_projected_audio_vectors, valid_projected_silence_vectors = self.extract_valid_vectors(
            batch_mask,
            batch_text_features,
            batch_vis_features,
            None,
            batch_silence_features=batch_silence_features,
        ) 
        
        moe_loss = None
        if self.config.fuse_type != "text_only":
            # print("batch_text_features:{}, batch_vis_features:{}, batch_audio_features: {}".format(batch_text_features, batch_vis_features, batch_audio_features))
            if self.config.cross_encoder_type == "ca" or self.config.cross_encoder_type == "ma":
                batch_text_features, batch_vis_features, _ = self.cross_encoder(batch_mask, batch_text_features, batch_vis_features, None)
            elif "moe" in self.config.cross_encoder_type:
                batch_text_features, batch_vis_features, _, moe_loss = self.cross_encoder(batch_mask, batch_text_features, batch_vis_features, None)
            else:
                raise ValueError("not support cross_encoder_type: {}".format(self.config.cross_encoder_type))

        valid_text_vectors, valid_visual_vectors, valid_audio_vectures, valid_silence_vectors = self.extract_valid_vectors(
            batch_mask,
            batch_text_features,
            batch_vis_features,
            None,
            batch_silence_features=batch_silence_features,
        ) 
        # 去batch
        silence_for_head = valid_silence_vectors if self.use_silence else None
        logits, loss = self.predictor(
            labels,
            valid_text_vectors,
            valid_visual_vectors,
            moe_loss,
            valid_projected_text_vectors,
            valid_projected_visual_vectors,
            silence_features=silence_for_head,
        )
        logits = self.restore_logits(labels, logits)

        output = (logits,)
        return ((loss,) + output) if loss is not None else output
