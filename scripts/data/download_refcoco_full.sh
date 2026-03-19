#!/usr/bin/env bash
set -euo pipefail

# Full RefCOCO setup under dataSets/coco:
# - COCO train2014 images
# - COCO 2014 annotations
# - RefCOCO / RefCOCO+ / RefCOCOg annotation packs
#
# Destination layout:
#   dataSets/coco/train2014/
#   dataSets/coco/annotations/
#   dataSets/coco/refcoco/
#   dataSets/coco/refcoco+/
#   dataSets/coco/refcocog/

ROOT="/home/iibrohimm/project/next_step/dataSets/coco"
DL_DIR="${ROOT}/downloads/refcoco"

mkdir -p "${DL_DIR}"
mkdir -p "${ROOT}"

echo "[start] $(date -Is)"
echo "[info] ROOT=${ROOT}"
echo "[info] DL_DIR=${DL_DIR}"

download() {
  local url="$1"
  local out="$2"
  if [[ -f "${out}" ]]; then
    echo "[skip] already exists: ${out}"
  else
    echo "[dl] ${url}"
  fi
  wget -c "${url}" -O "${out}"
}

download "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip" \
  "${DL_DIR}/refcoco.zip"
download "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip" \
  "${DL_DIR}/refcoco_plus.zip"
download "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip" \
  "${DL_DIR}/refcocog.zip"

download "http://images.cocodataset.org/zips/train2014.zip" \
  "${DL_DIR}/train2014.zip"
download "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" \
  "${DL_DIR}/annotations_trainval2014.zip"

echo "[extract] RefCOCO archives"
unzip -qo "${DL_DIR}/refcoco.zip" -d "${ROOT}"
unzip -qo "${DL_DIR}/refcoco_plus.zip" -d "${ROOT}"
unzip -qo "${DL_DIR}/refcocog.zip" -d "${ROOT}"

echo "[extract] COCO 2014 annotations"
unzip -qo "${DL_DIR}/annotations_trainval2014.zip" -d "${ROOT}"

if [[ -d "${ROOT}/train2014" ]]; then
  echo "[skip] train2014/ already exists"
else
  echo "[extract] COCO train2014 images (this takes time)"
  unzip -qo "${DL_DIR}/train2014.zip" -d "${ROOT}"
fi

echo "[done] $(date -Is)"
echo "[check] expected paths:"
echo "  - ${ROOT}/train2014"
echo "  - ${ROOT}/annotations/instances_train2014.json"
echo "  - ${ROOT}/refcoco/refs(unc).p"
echo "  - ${ROOT}/refcoco+/refs(unc).p"
echo "  - ${ROOT}/refcocog/refs(umd).p"
