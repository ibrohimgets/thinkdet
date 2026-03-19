"""
Flickr30k Entities Dataset for Visual Grounding & Reasoning

Referring expressions like "Two people talking" require understanding:
- Compositional phrases (adjectives, counts, relationships)
- Spatial context ("next to", "behind")
- Multi-object relationships

Better for reasoning than simple object detection.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_internvl_transform(image_size=448):
    """InternVL preprocessing."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_dino_transform():
    """GroundingDINO preprocessing."""
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    class GroundingDINOTransform:
        def __call__(self, img):
            # Keep aspect ratio
            w, h = img.size
            max_size = 800
            scale = min(max_size / max(w, h), 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            return normalize(img)
    
    return GroundingDINOTransform()


class Flickr30kGroundingDataset(Dataset):
    """
    Flickr30k Entities dataset for referring expression grounding.
    
    Returns per sample:
        internvl_image:  [3, 448, 448]
        dino_image:      [3, H, W]
        boxes:           [N, 4]  (cx, cy, w, h) normalized [0, 1]
        phrases:         List[str]  referring expressions per box
        query_text:      str  concatenated phrases "phrase1 . phrase2 ."
        image_id:        str
    """
    
    def __init__(self, annotations_file, img_dir=None, internvl_size=448, max_boxes=100):
        self.internvl_size = internvl_size
        self.max_boxes = max_boxes
        
        with open(annotations_file) as f:
            self.data = json.load(f)
        
        # Filter: only keep samples with boxes
        self.data = [item for item in self.data 
                     if any(not p.get('is_nobox', False) and p.get('boxes') 
                           for p in item.get('phrases', []))]
        
        # Override img_dir if provided, else use path in annotation
        self.img_dir = img_dir
        
        self.internvl_transform = build_internvl_transform(internvl_size)
        self.dino_transform = build_dino_transform()
        
        print(f"[Flickr30kGrounding] {len(self.data)} samples with referring expressions")
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def build_query_text(phrases):
        """Build GroundingDINO-style query from phrases."""
        unique = sorted(set(phrases))
        return ' . '.join(unique) + ' .'
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Image path
        if self.img_dir:
            img_path = os.path.join(self.img_dir, os.path.basename(item['image_path']))
        else:
            img_path = item['image_path']
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a placeholder
            return self.__getitem__((idx + 1) % len(self))
        
        img_w, img_h = img.size
        
        # Extract phrases and boxes
        boxes_list = []
        phrases_list = []
        
        for phrase_item in item['phrases']:
            if phrase_item.get('is_nobox', False):
                continue
            boxes = phrase_item.get('boxes', [])
            if not boxes:
                continue
            
            phrase = phrase_item['phrase']
            
            # Each phrase can have multiple boxes (e.g., "Two people")
            for box in boxes:
                x1, y1, x2, y2 = box  # absolute pixels
                # Convert to cx, cy, w, h normalized
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                
                # Clamp to [0, 1]
                cx = np.clip(cx, 0, 1)
                cy = np.clip(cy, 0, 1)
                w = np.clip(w, 0, 1)
                h = np.clip(h, 0, 1)
                
                if w > 0.01 and h > 0.01:  # valid box
                    boxes_list.append([cx, cy, w, h])
                    phrases_list.append(phrase)
        
        if len(boxes_list) == 0:
            # No valid boxes, return next sample
            return self.__getitem__((idx + 1) % len(self))
        
        # Limit to max_boxes
        if len(boxes_list) > self.max_boxes:
            boxes_list = boxes_list[:self.max_boxes]
            phrases_list = phrases_list[:self.max_boxes]
        
        boxes = torch.tensor(boxes_list, dtype=torch.float32)
        
        # Transform images
        internvl_img = self.internvl_transform(img)
        dino_img = self.dino_transform(img)
        
        query_text = self.build_query_text(phrases_list)
        
        return {
            'internvl_images': internvl_img,
            'dino_images': dino_img,
            'boxes': boxes,
            'phrases': phrases_list,
            'query_texts': query_text,
            'image_ids': item['image_id'],
        }


def collate_fn(batch):
    """Collate for DataLoader."""
    internvl_imgs = torch.stack([b['internvl_images'] for b in batch])
    
    # GroundingDINO: create NestedTensor
    dino_imgs = [b['dino_images'] for b in batch]
    max_h = max(img.shape[1] for img in dino_imgs)
    max_w = max(img.shape[2] for img in dino_imgs)
    
    batch_size = len(dino_imgs)
    padded = torch.zeros(batch_size, 3, max_h, max_w)
    mask = torch.ones(batch_size, max_h, max_w, dtype=torch.bool)
    
    for i, img in enumerate(dino_imgs):
        h, w = img.shape[1], img.shape[2]
        padded[i, :, :h, :w] = img
        mask[i, :h, :w] = False
    
    from groundingdino.util.misc import NestedTensor
    dino_nested = NestedTensor(padded, mask)
    
    return {
        'internvl_images': internvl_imgs,
        'dino_images': dino_nested,
        'boxes': [b['boxes'] for b in batch],
        'phrases': [b['phrases'] for b in batch],
        'query_texts': [b['query_texts'] for b in batch],
        'image_ids': [b['image_ids'] for b in batch],
    }


# ── Positive map builder for phrases ──
def build_phrase_positive_map(tokenizer, special_tokens, all_phrases, max_text_len=512):
    """
    Build positive map for phrase-based grounding.
    Same logic as COCO build_positive_map but for phrases instead of categories.
    
    Args:
        tokenizer: GroundingDINO tokenizer
        special_tokens: Special token IDs (for finding separators)
        all_phrases: List of unique phrases
        max_text_len: Max token length (default 512)
    
    Returns:
        query_text: Concatenated query string
        positive_map: [num_phrases, max_text_len] binary
        positive_map_norm: [num_phrases, max_text_len] normalized
        phrase_to_idx: Dict mapping phrase → index
    """
    # Limit number of phrases to avoid token overflow (512 token limit)
    # Each phrase takes ~5-10 tokens, so 200 phrases ~ 1000-2000 tokens (too many!)
    # Use top 100 for safety
    MAX_PHRASES = 100
    if len(all_phrases) > MAX_PHRASES:
        # Take first N sorted phrases for determinism
        all_phrases = sorted(set(all_phrases))[:MAX_PHRASES]
    else:
        all_phrases = sorted(set(all_phrases))
    
    query_text = ' . '.join(all_phrases) + ' .'
    tokenized = tokenizer(query_text, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]
    
    special_set = set(special_tokens)
    sep_positions = []
    for pos, tid in enumerate(input_ids.tolist()):
        if tid in special_set:
            sep_positions.append(pos)
    
    positive_map = torch.zeros(len(all_phrases), max_text_len)
    phrase_idx = 0
    for i in range(len(sep_positions) - 1):
        start = sep_positions[i] + 1
        end = sep_positions[i + 1]
        if start < end and phrase_idx < len(all_phrases):
            positive_map[phrase_idx, start:end] = 1.0
            phrase_idx += 1
    
    row_sums = positive_map.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    positive_map_norm = positive_map / row_sums
    
    phrase_to_idx = {p: i for i, p in enumerate(all_phrases)}
    
    return query_text, positive_map, positive_map_norm, phrase_to_idx
