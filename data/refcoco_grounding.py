"""
ThinkDet v2 - RefCOCO/RefCOCO+/RefCOCOg Dataset for REC

Each sample yields ONE (image, referring_expression, bounding_box) triple.
During training, randomly selects one sentence per ref.
During eval, expands all sentences so each is a separate sample.

Produces paired inputs for InternVL + GroundingDINO:
    - InternVL:  [3, 448, 448] + expression string
    - DINO:      NestedTensor  + "expression ." caption
    - GT:        [1, 4] box (cx, cy, w, h) normalized
"""

import os
import json
import random
import pickle
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

import sys
sys.path.insert(0, '/home/iibrohimm/project/next_step/GroundingDINO/GroundingDINO')
import groundingdino.datasets.transforms as GD_T
from groundingdino.util.misc import nested_tensor_from_tensor_list

import torchvision.transforms as TV_T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_internvl_transform(input_size=448):
    return TV_T.Compose([
        TV_T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        TV_T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        TV_T.ToTensor(),
        TV_T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_dino_transform():
    return GD_T.Compose([
        GD_T.RandomResize([800], max_size=1333),
        GD_T.ToTensor(),
        GD_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class RefCOCOGroundingDataset(Dataset):
    """
    RefCOCO/RefCOCO+/RefCOCOg dataset for Referring Expression Comprehension.

    Returns per sample:
        internvl_image:  [3, 448, 448]
        dino_image:      [3, H, W] (variable, collated into NestedTensor)
        box:             [1, 4] (cx, cy, w, h) normalized [0, 1]
        expression:      str  - the referring expression
        query_text:      str  - "expression ." for GroundingDINO
        ref_id:          int
        image_id:        int
    """

    def __init__(
        self,
        data_root,
        dataset_name,
        split_by,
        split,
        image_dir,
        internvl_size=448,
        train_mode=True,
        use_hflip=False,
    ):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.split = split
        self.image_dir = image_dir
        self.train_mode = train_mode
        self.use_hflip = use_hflip
        self.internvl_size = internvl_size

        # Load refs (Python 3 compatible)
        ref_file = os.path.join(data_root, dataset_name, f'refs({split_by}).p')
        with open(ref_file, 'rb') as f:
            all_refs = pickle.load(f, encoding='latin1')

        # Load instances (COCO format)
        inst_file = os.path.join(data_root, dataset_name, 'instances.json')
        with open(inst_file, 'r') as f:
            instances = json.load(f)

        # Build indexes
        self.ann_map = {a['id']: a for a in instances['annotations']}
        self.img_map = {i['id']: i for i in instances['images']}

        # Filter refs by split
        if split in ['testA', 'testB']:
            split_refs = [r for r in all_refs if split[-1] in r['split']]
        elif split == 'test':
            split_refs = [r for r in all_refs if 'test' in r['split']]
        else:
            split_refs = [r for r in all_refs if r['split'] == split]

        # Build samples
        if train_mode:
            # One entry per ref; sentence chosen randomly in __getitem__
            self.samples = []
            for ref in split_refs:
                ann = self.ann_map.get(ref['ann_id'])
                if ann is None or ann.get('area', 1) <= 0:
                    continue
                self.samples.append(ref)
        else:
            # Expand: one entry per (ref, sentence) for comprehensive eval
            self.samples = []
            for ref in split_refs:
                ann = self.ann_map.get(ref['ann_id'])
                if ann is None or ann.get('area', 1) <= 0:
                    continue
                for sent in ref['sentences']:
                    self.samples.append({
                        'ref': ref,
                        'sent': sent['sent'],
                    })

        self.internvl_transform = build_internvl_transform(internvl_size)
        self.dino_transform = build_dino_transform()

        print(f"[RefCOCO] {dataset_name}/{split}({split_by}): "
              f"{len(self.samples)} samples ({'train' if train_mode else 'eval'} mode)")

    def __len__(self):
        return len(self.samples)

    def _get_image_path(self, image_id):
        return os.path.join(self.image_dir, f'{image_id:012d}.jpg')

    def __getitem__(self, idx):
        if self.train_mode:
            ref = self.samples[idx]
            expression = random.choice(ref['sentences'])['sent']
        else:
            ref = self.samples[idx]['ref']
            expression = self.samples[idx]['sent']

        ann = self.ann_map[ref['ann_id']]
        image_id = ref['image_id']
        img_info = self.img_map[image_id]
        img_w, img_h = img_info['width'], img_info['height']

        img_path = self._get_image_path(image_id)
        pil_image = Image.open(img_path).convert('RGB')

        # Box: [x, y, w, h] absolute → (cx, cy, w, h) normalized
        x, y, w, h = ann['bbox']
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        nw = w / img_w
        nh = h / img_h

        # Optional horizontal flip (disabled by default for REC because
        # directional words like "left"/"right" become inconsistent).
        do_flip = self.train_mode and self.use_hflip and random.random() < 0.5
        if do_flip:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            cx = 1.0 - cx

        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(1e-4, min(1.0, nw))
        nh = max(1e-4, min(1.0, nh))

        box = torch.tensor([[cx, cy, nw, nh]], dtype=torch.float32)

        internvl_image = self.internvl_transform(pil_image)
        dino_image, _ = self.dino_transform(pil_image, None)

        query_text = expression.strip().lower() + ' .'

        return {
            'internvl_image': internvl_image,
            'dino_image': dino_image,
            'box': box,
            'expression': expression,
            'query_text': query_text,
            'ref_id': ref['ref_id'],
            'image_id': image_id,
        }


def refcoco_collate_fn(batch):
    internvl_images = torch.stack([s['internvl_image'] for s in batch])
    dino_images = nested_tensor_from_tensor_list(
        [s['dino_image'] for s in batch]
    )
    return {
        'internvl_images': internvl_images,
        'dino_images': dino_images,
        'boxes': [s['box'] for s in batch],
        'expressions': [s['expression'] for s in batch],
        'query_texts': [s['query_text'] for s in batch],
        'ref_ids': [s['ref_id'] for s in batch],
        'image_ids': [s['image_id'] for s in batch],
    }


# ── Positive map builder for REC ──

def build_rec_positive_map_batch(tokenizer, special_tokens, expressions,
                                  max_text_len=512):
    """
    Build per-sample positive maps for referring expressions.

    For expression "the person on the left", query = "the person on the left ."
    All non-special tokens are marked positive for the single GT box.

    Args:
        tokenizer: GroundingDINO BERT tokenizer
        special_tokens: List of special token IDs ([CLS], [SEP], ., ?)
        expressions: List[str] referring expressions (B items)

    Returns:
        query_texts:       List[str] formatted as "expr ."
        positive_maps:     [B, 1, max_text_len] binary
        positive_maps_norm: [B, 1, max_text_len] row-normalized
    """
    B = len(expressions)
    query_texts = [e.strip().lower() + ' .' for e in expressions]

    positive_maps = torch.zeros(B, 1, max_text_len)
    special_set = set(special_tokens)

    for b, qt in enumerate(query_texts):
        tokenized = tokenizer(qt, return_tensors="pt")
        input_ids = tokenized["input_ids"][0]
        for pos, tid in enumerate(input_ids.tolist()):
            if pos < max_text_len and tid not in special_set:
                positive_maps[b, 0, pos] = 1.0

    row_sums = positive_maps.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    positive_maps_norm = positive_maps / row_sums

    return query_texts, positive_maps, positive_maps_norm


# ── Joint multi-dataset builder ──

def build_joint_refcoco_train(data_root, image_dir, internvl_size=448):
    datasets = []
    configs = [
        ('refcoco',  'unc'),
        ('refcoco+', 'unc'),
        ('refcocog', 'umd'),
    ]
    for name, split_by in configs:
        ds = RefCOCOGroundingDataset(
            data_root=data_root,
            dataset_name=name,
            split_by=split_by,
            split='train',
            image_dir=image_dir,
            internvl_size=internvl_size,
            train_mode=True,
            use_hflip=False,
        )
        datasets.append(ds)
    combined = ConcatDataset(datasets)
    print(f"[RefCOCO] Joint train: {len(combined)} total refs")
    return combined


def build_refcoco_eval(data_root, image_dir, dataset_name, split_by, split,
                        internvl_size=448):
    return RefCOCOGroundingDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        split_by=split_by,
        split=split,
        image_dir=image_dir,
        internvl_size=internvl_size,
        train_mode=False,
    )
