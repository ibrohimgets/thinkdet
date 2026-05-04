"""
ThinkDet v2 - InternVL Feature Extractor

Uses InternVL native forward with [image + text] jointly, then extracts
the text-conditioned visual hidden states as H_vlm.

    v1: H_vlm = Qwen2(vision_tokens)               -> query-independent
    v2: H_vlm = InternVL(vision_tokens + text)     -> query-conditioned
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from typing import Optional, List, Tuple
import math


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_internvl_transform(input_size=448):
    """Build the standard InternVL image transform."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


class InternVLFeatureExtractor(nn.Module):
    """
    Extracts text-conditioned visual features (H_vlm) from InternVL.

    Tested with:
        - InternVL2-1B   (Qwen2-0.5B,  hidden=896,  24 layers)
        - InternVL3.5-1B (Qwen3-0.6B,  hidden=1024, 28 layers)  <-- default

    Pipeline:
        Image -> InternViT -> pixel_shuffle -> MLP -> [B, 256, D]
        Text  -> tokenizer -> embed_tokens   ->       [B, T, D]
        Concat -> LLM layers 0..N -> extract visual positions -> H_vlm

    Args:
        model_path: Path to InternVL local directory or HuggingFace ID
        extract_layer: Single LLM layer to extract from (default 8)
        extract_layers: Optional list of layers to fuse. If provided,
                        takes priority over extract_layer.
        layer_fusion: Fusion strategy for extract_layers. Supported:
                      - 'mean': average selected layer features
                      - 'last': use only the highest selected layer
                      - 'learned': softmax-normalized global layer weights
        max_text_len: Max text tokens for detection queries
        freeze: Whether to freeze all of InternVL
        use_flash_attn: Whether to use flash attention
    """

    def __init__(
        self,
        model_path: str,
        extract_layer: int = 8,
        extract_layers: Optional[List[int]] = None,
        layer_fusion: str = 'mean',
        max_text_len: int = 256,
        freeze: bool = True,
        use_flash_attn: bool = False,
        use_official_prompt_extraction: bool = False,
    ):
        super().__init__()

        self.model_path = model_path
        self.layer_fusion = layer_fusion
        self.use_official_prompt_extraction = bool(use_official_prompt_extraction)
        self.extract_layers = self._canonicalize_extract_layers(
            extract_layer=extract_layer,
            extract_layers=extract_layers,
        )
        self.extract_layer = max(self.extract_layers)  # backward-compatible alias
        self.max_text_len = max_text_len

        self._load_model(model_path, use_flash_attn)

        # Read hidden dim from the loaded model -- works for any InternVL variant
        self.llm_hidden_dim = self.internvl.language_model.config.hidden_size
        if len(self.extract_layers) > 1 and self.layer_fusion == 'learned':
            self.layer_fusion_logits = torch.nn.Parameter(
                torch.zeros(len(self.extract_layers))
            )
        else:
            self.register_parameter("layer_fusion_logits", None)

        if freeze:
            self.freeze()

        if len(self.extract_layers) == 1:
            layer_desc = f"{self.extract_layers[0]}"
        else:
            layer_desc = f"{self.extract_layers} (fusion={self.layer_fusion})"
        print(
            f"[ThinkDet v2] InternVL loaded | layer={layer_desc}/{self.num_llm_layers} | "
            f"vis_tokens={self.num_image_token} | dim={self.llm_hidden_dim} | "
            f"official_extraction={self.use_official_prompt_extraction}"
        )

    @staticmethod
    def _canonicalize_extract_layers(extract_layer, extract_layers):
        if extract_layers is None:
            return [int(extract_layer)]

        if not isinstance(extract_layers, (list, tuple)):
            raise ValueError("extract_layers must be a list or tuple of ints")
        if len(extract_layers) == 0:
            raise ValueError("extract_layers cannot be empty")

        return sorted(set(int(x) for x in extract_layers))

    def _load_model(self, model_path, use_flash_attn):
        from transformers import AutoTokenizer, AutoModel

        self.internvl = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            use_flash_attn=use_flash_attn,
            low_cpu_mem_usage=False,
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
                fix_mistral_regex=True,
            )
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.num_image_token = self.internvl.num_image_token  # 256
        self.num_llm_layers = self.internvl.language_model.config.num_hidden_layers  # 24 or 28

        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.internvl.img_context_token_id = self.img_context_token_id

        clamped = []
        for layer_idx in self.extract_layers:
            if layer_idx < 0:
                raise ValueError(f"extract layer must be non-negative, got {layer_idx}")
            if layer_idx >= self.num_llm_layers:
                layer_idx = self.num_llm_layers - 1
            clamped.append(layer_idx)

        self.extract_layers = sorted(set(clamped))
        self.extract_layer = max(self.extract_layers)

        if self.layer_fusion not in ('mean', 'last', 'learned'):
            raise ValueError(
                f"Unsupported layer_fusion '{self.layer_fusion}'. Use 'mean', 'last', or 'learned'."
            )

    def freeze(self):
        for param in self.internvl.parameters():
            param.requires_grad = False

    def unfreeze_last_n_llm_layers(self, n=2):
        total = self.num_llm_layers
        for i, layer in enumerate(self.internvl.language_model.model.layers):
            if i >= total - n:
                for param in layer.parameters():
                    param.requires_grad = True
        count = sum(p.numel() for p in self.internvl.parameters() if p.requires_grad)
        print(f"[ThinkDet v2] Unfroze last {n} LLM layers ({count:,} trainable params)")

    def _build_input_embeds(self, pixel_values, text_queries):
        """Build joint [visual + text] embeddings for the LLM."""
        device = pixel_values.device
        B = pixel_values.shape[0]

        # Visual: InternViT -> pixel_shuffle -> MLP -> [B, 256, D]
        vit_embeds = self.internvl.extract_feature(pixel_values)
        num_vis = vit_embeds.shape[1]

        # Text: tokenize -> embed -> [B, T, D]
        text_inputs = self.tokenizer(
            text_queries,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
        ).to(device)

        text_embeds = self.internvl.language_model.get_input_embeddings()(
            text_inputs.input_ids
        )

        # Concat [visual ; text]
        input_embeds = torch.cat([vit_embeds, text_embeds], dim=1)

        # Attention mask
        vis_mask = torch.ones(B, num_vis, device=device, dtype=torch.long)
        attention_mask = torch.cat([vis_mask, text_inputs.attention_mask], dim=1)

        return input_embeds, attention_mask, num_vis

    def _forward_selected_layers(self, input_embeds, attention_mask, num_vis):
        """
        Forward through InternVL LLM up to max selected layer and return
        visual-token features for each selected layer.
        """
        B, seq_len, _ = input_embeds.shape
        device = input_embeds.device
        dtype = input_embeds.dtype

        llm = self.internvl.language_model.model
        hidden_states = input_embeds

        # Bidirectional extraction mask (only pad tokens masked).
        pad_mask = attention_mask[:, None, None, :].to(dtype)      # [B,1,1,S]
        attn_mask = (1.0 - pad_mask) * torch.finfo(dtype).min      # 0 / -inf

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)

        position_embeddings = None
        if hasattr(llm, 'rotary_emb'):
            position_embeddings = llm.rotary_emb(hidden_states, position_ids)

        target_layers = set(self.extract_layers)
        max_layer = max(self.extract_layers)
        selected_features = {}

        for i in range(max_layer + 1):
            layer_out = llm.layers[i](
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            if i in target_layers:
                selected_features[i] = hidden_states[:, :num_vis, :]

        return selected_features

    def _build_official_queries(self, text_queries, num_patches=1):
        queries = []
        for text_query in text_queries:
            question = text_query if '<image>' in text_query else '<image>\n' + text_query
            if hasattr(self.internvl, 'conv_template'):
                template = self.internvl.conv_template.copy()
                template.system_message = getattr(
                    self.internvl,
                    'system_message',
                    template.system_message,
                )
            else:
                from conversation import get_conv_template
                template = get_conv_template(self.internvl.template)
                template.system_message = getattr(
                    self.internvl,
                    'system_message',
                    template.system_message,
                )
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            image_tokens = (
                '<img>'
                + '<IMG_CONTEXT>' * self.num_image_token * int(num_patches)
                + '</img>'
            )
            queries.append(query.replace('<image>', image_tokens, 1))
        return queries

    def _extract_selected_layers_official(self, pixel_values, text_queries):
        """
        Official InternVL chat-style extraction.

        Visual embeddings are inserted at <IMG_CONTEXT> token positions inside
        the tokenized chat prompt. Returned features are the hidden states at
        those image-token positions for each selected Qwen layer.
        """
        device = pixel_values.device
        B = pixel_values.shape[0]
        vit_embeds = self.internvl.extract_feature(pixel_values)
        num_vis = vit_embeds.shape[1]

        if len(text_queries) != B:
            raise ValueError(f"text_queries length {len(text_queries)} != batch size {B}")

        queries = self._build_official_queries(text_queries, num_patches=1)
        model_inputs = self.tokenizer(
            queries,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_text_len + self.num_image_token + 64,
        ).to(device)

        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        image_mask = input_ids == self.img_context_token_id
        expected = B * num_vis
        actual = int(image_mask.sum().item())
        if actual != expected:
            raise RuntimeError(
                f"<IMG_CONTEXT> token count mismatch: tokenizer={actual}, "
                f"visual_features={expected}"
            )

        input_embeds = self.internvl.language_model.get_input_embeddings()(input_ids).clone()
        _, seq_len, hidden_dim = input_embeds.shape
        flat_embeds = input_embeds.reshape(B * seq_len, hidden_dim)
        flat_mask = image_mask.reshape(B * seq_len)
        flat_embeds[flat_mask] = vit_embeds.reshape(-1, hidden_dim).to(flat_embeds.dtype)
        input_embeds = flat_embeds.reshape(B, seq_len, hidden_dim)

        outputs = self.internvl.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        selected_features = {}
        for layer_idx in self.extract_layers:
            hidden_states = outputs.hidden_states[layer_idx + 1]
            image_features = hidden_states[image_mask].reshape(B, num_vis, hidden_dim)
            selected_features[layer_idx] = image_features
        return selected_features

    def extract_selected_layers(self, pixel_values, text_queries):
        """
        Extract selected InternVL layers for the given image/text batch.

        Returns:
            dict[layer_idx] -> [B, 256, D]
        """
        if self.use_official_prompt_extraction:
            return self._extract_selected_layers_official(pixel_values, text_queries)

        input_embeds, attention_mask, num_vis = self._build_input_embeds(
            pixel_values, text_queries
        )
        return self._forward_selected_layers(input_embeds, attention_mask, num_vis)

    def fuse_selected_layers(self, selected_features, weights=None):
        """
        Fuse selected layer features.

        Args:
            selected_features: dict[layer_idx] -> [B, 256, D]
            weights:
                - None: simple mean or learned global weights over selected layers
                - [B, L]: per-sample layer weights (L = len(self.extract_layers))
                - [L]: global layer weights
        Returns:
            H_vlm: [B, 256, D]
        """
        if len(self.extract_layers) == 1:
            return selected_features[self.extract_layer]

        feats = [selected_features[i] for i in self.extract_layers]
        stack = torch.stack(feats, dim=1)  # [B, L, 256, D]

        if weights is None:
            if self.layer_fusion == 'learned':
                if self.layer_fusion_logits is None:
                    raise RuntimeError("learned layer fusion requested without logits")
                weights = torch.softmax(self.layer_fusion_logits, dim=0)
            else:
                return stack.mean(dim=1)

        if weights.dim() == 1:
            w = weights.unsqueeze(0)  # [1, L]
        elif weights.dim() == 2:
            w = weights
        else:
            raise ValueError(f"weights must have dim 1 or 2, got {weights.dim()}")

        if w.shape[-1] != len(self.extract_layers):
            raise ValueError(
                f"weights last dim {w.shape[-1]} != num extract layers {len(self.extract_layers)}"
            )

        w = w.to(stack.device, dtype=stack.dtype)
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        w = w / w_sum

        return (stack * w[:, :, None, None]).sum(dim=1)

    def forward(self, pixel_values, text_queries):
        """
        Args:
            pixel_values: [B, 3, 448, 448]
            text_queries: List[str] e.g. ["person . car .", "dog . cat ."]

        Returns:
            H_vlm: [B, 256, D] text-conditioned visual features
                   D = 1024 for InternVL3.5-1B, 896 for InternVL2-1B
        """
        selected_features = self.extract_selected_layers(pixel_values, text_queries)

        if self.layer_fusion == 'last' or len(self.extract_layers) == 1:
            return selected_features[self.extract_layer]

        return self.fuse_selected_layers(selected_features, weights=None)

    def get_trainable_params(self):
        params = [p for p in self.internvl.parameters() if p.requires_grad]
        params.extend([p for p in self.get_fusion_params() if p.requires_grad])
        return params

    def get_fusion_params(self):
        if self.layer_fusion_logits is None:
            return []
        return [self.layer_fusion_logits]
