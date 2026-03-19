"""
ThinkDet v2 - COCO Grounding Dataset

Produces paired inputs for both InternVL and GroundingDINO:
    - InternVL:  PIL image → [3, 448, 448]  + text query string
    - DINO:      PIL image → NestedTensor (resized to 800, max 1333)
    - GT:        boxes (cxcywh, normalized) + category names

Each sample groups all categories present in the image into a single
GroundingDINO-style query: "person . dog . car ."

The positive_map links each category name to its BERT token positions
so the contrastive classification loss knows which logit columns
correspond to which GT box.
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
import numpy as np

# GroundingDINO transforms
import sys
sys.path.insert(0, '/home/iibrohimm/project/next_step/GroundingDINO/GroundingDINO')
import groundingdino.datasets.transforms as GD_T
from groundingdino.util.misc import NestedTensor, nested_tensor_from_tensor_list

# InternVL transform
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


class COCOGroundingDataset(Dataset):
    """
    COCO dataset that yields paired InternVL + GroundingDINO inputs.

    Query modes:
        - fixed: always use full 80-class fixed vocabulary query
        - gt:    use per-image GT category query
        - mixed: stochastic mix of fixed and GT queries

    Returns per sample:
        internvl_image:  [3, 448, 448]
        dino_image:      [3, H, W]   (variable, collated into NestedTensor)
        boxes:           [N, 4]  (cx, cy, w, h) normalized to [0, 1]
        category_names:  List[str]  per-box category names
        query_text:      str  "cat1 . cat2 . cat3 ."
        image_id:        int
    """

    def __init__(
        self,
        img_dir,
        ann_file,
        internvl_size=448,
        max_boxes=100,
        query_mode="fixed",
        dynamic_query_ratio=0.7,
        max_query_categories=12,
    ):
        self.img_dir = img_dir
        self.internvl_size = internvl_size
        self.max_boxes = max_boxes
        self.query_mode = str(query_mode).lower()
        self.dynamic_query_ratio = float(dynamic_query_ratio)
        self.max_query_categories = int(max_query_categories)
        if self.query_mode not in {"fixed", "gt", "mixed"}:
            raise ValueError(
                f"Unsupported query_mode={query_mode}. "
                "Choose from {'fixed','gt','mixed'}."
            )

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cats = self.coco.loadCats(self.cat_ids)
        self.cat_id_to_name = {c['id']: c['name'] for c in self.cats}
        self.all_cat_names_sorted = sorted([self.cat_id_to_name[cid] for cid in self.cat_ids])
        self.fixed_vocabulary_query = ' . '.join(self.all_cat_names_sorted) + ' .'

        # Only keep images that have at least one valid annotation
        all_img_ids = sorted(self.coco.getImgIds())
        self.img_ids = []
        for img_id in all_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            if len(ann_ids) > 0:
                self.img_ids.append(img_id)

        self.internvl_transform = build_internvl_transform(internvl_size)
        self.dino_transform = build_dino_transform()

        print(
            f"[COCOGrounding] {len(self.img_ids)} images, "
            f"{len(self.cat_ids)} categories from {ann_file} "
            f"(query_mode={self.query_mode})"
        )

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def build_query_text(category_names, max_categories=0, shuffle=True):
        """Build GroundingDINO-style query: 'person . dog . car .'"""
        unique = list(set(category_names))
        if shuffle:
            random.shuffle(unique)
        else:
            unique = sorted(unique)
        if max_categories and max_categories > 0 and len(unique) > max_categories:
            unique = unique[:max_categories]
        return ' . '.join(unique) + ' .'

    def build_fixed_vocabulary_query(self):
        """Build query with ALL 80 COCO categories (no GT leakage)"""
        return self.fixed_vocabulary_query

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        pil_image = Image.open(img_path).convert('RGB')

        img_w, img_h = img_info['width'], img_info['height']

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        category_names = []

        for ann in anns:
            if ann['area'] <= 0:
                continue
            x, y, w, h = ann['bbox']
            # Convert to normalized cxcywh
            cx = (x + w / 2.0) / img_w
            cy = (y + h / 2.0) / img_h
            nw = w / img_w
            nh = h / img_h
            # Clamp to valid range
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(1e-4, min(1.0, nw))
            nh = max(1e-4, min(1.0, nh))
            boxes.append([cx, cy, nw, nh])
            category_names.append(self.cat_id_to_name[ann['category_id']])

        # Limit boxes
        if len(boxes) > self.max_boxes:
            indices = np.random.choice(len(boxes), self.max_boxes, replace=False)
            boxes = [boxes[i] for i in indices]
            category_names = [category_names[i] for i in indices]

        # Build query text for InternVL conditioning.
        # DINO caption text can still be provided externally by the training script.
        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)

        if self.query_mode == "fixed":
            query_text = self.build_fixed_vocabulary_query()
        elif self.query_mode == "gt":
            query_text = self.build_query_text(
                category_names,
                max_categories=self.max_query_categories,
                shuffle=True,
            ) if len(category_names) > 0 else self.build_fixed_vocabulary_query()
        else:  # mixed
            use_dynamic = (
                len(category_names) > 0
                and random.random() < self.dynamic_query_ratio
            )
            if use_dynamic:
                query_text = self.build_query_text(
                    category_names,
                    max_categories=self.max_query_categories,
                    shuffle=True,
                )
            else:
                query_text = self.build_fixed_vocabulary_query()

        # InternVL image
        internvl_image = self.internvl_transform(pil_image)  # [3, 448, 448]

        # GroundingDINO image
        dino_image, _ = self.dino_transform(pil_image, None)  # [3, H, W]

        return {
            'internvl_image': internvl_image,
            'dino_image': dino_image,
            'boxes': boxes_t,
            'category_names': category_names,
            'query_text': query_text,
            'image_id': img_id,
        }


def collate_fn(batch):
    """
    Custom collate that:
      - Stacks InternVL images into [B, 3, 448, 448]
      - Wraps DINO images into NestedTensor (handles variable sizes)
      - Keeps boxes/categories as lists (variable count per image)
    """
    internvl_images = torch.stack([s['internvl_image'] for s in batch])  # [B, 3, 448, 448]

    dino_images = nested_tensor_from_tensor_list(
        [s['dino_image'] for s in batch]
    )

    boxes = [s['boxes'] for s in batch]
    category_names = [s['category_names'] for s in batch]
    query_texts = [s['query_text'] for s in batch]
    image_ids = [s['image_id'] for s in batch]

    return {
        'internvl_images': internvl_images,
        'dino_images': dino_images,
        'boxes': boxes,
        'category_names': category_names,
        'query_texts': query_texts,
        'image_ids': image_ids,
    }


def build_positive_map(tokenizer, special_tokens, cat_names, max_text_len=512):
    """
    Build mapping from category index → BERT token positions.

    For query "person . dog . car .", maps:
        cat 0 ("car")    → token positions for "car"
        cat 1 ("dog")    → token positions for "dog"
        cat 2 ("person") → token positions for "person"

    Returns:
        query_text:       str
        positive_map:     [num_cats, max_text_len]  binary
        positive_map_norm: [num_cats, max_text_len]  row-normalized
        cat_name_to_idx:  dict  name → index in positive_map
    """
    query_text = ' . '.join(cat_names) + ' .'
    tokenized = tokenizer(query_text, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]

    special_set = set(special_tokens)
    sep_positions = []
    for pos, tid in enumerate(input_ids.tolist()):
        if tid in special_set:
            sep_positions.append(pos)

    positive_map = torch.zeros(len(cat_names), max_text_len)
    cat_idx = 0
    for i in range(len(sep_positions) - 1):
        start = sep_positions[i] + 1
        end = sep_positions[i + 1]
        if start < end and cat_idx < len(cat_names):
            positive_map[cat_idx, start:end] = 1.0
            cat_idx += 1

    row_sums = positive_map.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    positive_map_norm = positive_map / row_sums

    cat_name_to_idx = {name: idx for idx, name in enumerate(cat_names)}

    return query_text, positive_map, positive_map_norm, cat_name_to_idx


def build_coco_dataloader(
    img_dir,
    ann_file,
    batch_size=2,
    num_workers=4,
    shuffle=True,
    internvl_size=448,
):
    """Build the COCO grounding dataloader."""
    dataset = COCOGroundingDataset(
        img_dir=img_dir,
        ann_file=ann_file,
        internvl_size=internvl_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    return dataset, loader
