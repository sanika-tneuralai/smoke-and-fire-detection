# utils.py
import json
import os
from pathlib import Path

def write_data_yaml(train_dir, val_dir, out_path='data_smoke_fire.yaml', names=['smoke','fire']):
    d = {'train': str(Path(train_dir).resolve()), 'val': str(Path(val_dir).resolve()), 'nc': len(names), 'names': names}
    with open(out_path, 'w') as f:
        json.dump(d, f, indent=2)
    print("Wrote", out_path)

# If your Kaggle dataset uses COCO annotations (annotations.json), you'll need a converter
# Skeleton below — fill in per your annotation format.
def coco_to_yolo(coco_json_path, images_dir, out_labels_dir, names=['smoke','fire']):
    """
    Convert COCO-style JSON to YOLO txt files.
    This is a minimal skeleton — adapt category id / names mapping for your dataset.
    """
    import json
    from pathlib import Path
    Path(out_labels_dir).mkdir(parents=True, exist_ok=True)
    with open(coco_json_path) as f:
        coco = json.load(f)
    imgs = {img['id']: img for img in coco['images']}
    # map category id to 0..nc-1
    cat_map = {c['id']: i for i,c in enumerate(coco['categories'])}
    # build annotations per image
    ann_map = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        bbox = ann['bbox']  # [x,y,w,h] in pixels
        cat_id = ann['category_id']
        img = imgs[img_id]
        w_img, h_img = img['width'], img['height']
        # convert to YOLO: x_center y_center w h (normalized)
        x, y, w, h = bbox
        x_c = x + w/2
        y_c = y + h/2
        x_c /= w_img
        y_c /= h_img
        w /= w_img
        h /= h_img
        yolo_line = f"{cat_map[cat_id]} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n"
        ann_map.setdefault(img['file_name'], []).append(yolo_line)

    for fname, anns in ann_map.items():
        label_fname = Path(out_labels_dir) / (Path(fname).stem + ".txt")
        with open(label_fname, 'w') as f:
            f.writelines(anns)
    print("Converted COCO->YOLO for", len(ann_map), "images")
