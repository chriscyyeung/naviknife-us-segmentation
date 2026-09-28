import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw
from mmdet.apis import init_detector, inference_detector
from endosam.models.utils.custom_functional import multi_head_attention_forward

# Monkey patch the multi_head_attention_forward function
F.multi_head_attention_forward = multi_head_attention_forward


def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def extract_prompts(result):
    """Pull the box/point prompts that were actually fed to the model out of
    the returned data sample, and convert them back to ORIGINAL image pixel
    coordinates so they can be drawn on the input image.

    Prompts are stored in resized-image pixel space (original coords *
    scale_factor + 0.5, per the test-time branch of GetFullSizeBox /
    GetPointBox), so we undo that here.
    """
    gt = result.gt_instances
    meta = result.metainfo
    print("gt_instances fields:", list(gt.keys()))
    print(f"metainfo: ori_shape={meta.get('ori_shape')}, "
          f"img_shape={meta.get('img_shape')}, "
          f"pad_shape={meta.get('pad_shape')}, "
          f"scale_factor={meta.get('scale_factor')}")
    sx, sy = meta["scale_factor"]

    def to_original(xy):
        out = xy.astype(np.float64).copy()
        out[..., 0] = (out[..., 0] - 0.5) / sx
        out[..., 1] = (out[..., 1] - 0.5) / sy
        return out

    boxes = points = None
    if "boxes" in gt:
        raw = to_numpy(gt.boxes).reshape(-1, 2, 2)      # (N, 2 corners, xy)
        print("RAW box prompt(s) as given to the model:\n", raw)
        boxes = to_original(raw)
    if "points" in gt:
        raw = to_numpy(gt.points).reshape(-1, 2)         # (N, xy)
        print("RAW point prompt(s) as given to the model:\n", raw)
        points = to_original(raw)
    return boxes, points


def draw_prompts(base_img, boxes, points):
    vis = Image.fromarray(base_img.copy())
    draw = ImageDraw.Draw(vis)
    if boxes is not None:
        for (x1, y1), (x2, y2) in boxes:
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
    if points is not None:
        for x, y in points:
            r = 6
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=(255, 255, 0), outline=(0, 0, 0))
    return vis


def main():
    config_path = "/home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/external/UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_inference.py"
    ckpt_path = "/home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/external/UltraSam/work_dirs/UltraSAM_box_refine_fold0/iter_30000.pth"

    # Initialize the detector
    model = init_detector(config_path, ckpt_path, device='cuda:0')
    print(model.cfg.pretty_text)

    # Load an image
    img = Image.open("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/images/LN003_frame00000.png").convert("RGB")
    arr = np.array(img)
    print(arr.shape)
    img_h, img_w = arr.shape[:2]

    # Perform inference
    result = inference_detector(model, arr)

    # ---- Prompt sanity check ------------------------------------------
    boxes, points = extract_prompts(result)

    if boxes is not None:
        print(f"\nImage size (W x H): {img_w} x {img_h}")
        for i, ((x1, y1), (x2, y2)) in enumerate(boxes):
            coverage = max(x2 - x1, 0) * max(y2 - y1, 0) / (img_w * img_h)
            print(f"Box prompt {i} in original-image coords: "
                  f"({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})  "
                  f"covers {coverage:.1%} of the image")
            if coverage < 0.5:
                print("  [WARNING] This 'full-size' box covers less than half "
                      "the image -- the prompt itself looks wrong, so poor "
                      "segmentations may not be the model's fault.")
            elif coverage > 1.1:
                print("  [WARNING] This 'full-size' box is much LARGER than the "
                      "image (extends past its edges). A coverage near "
                      f"{(result.metainfo['scale_factor'][0] * result.metainfo['scale_factor'][1]):.2f}x "
                      "would mean the scale factor was applied twice.")
    if points is not None:
        for i, (x, y) in enumerate(points):
            print(f"Point prompt {i} in original-image coords: ({x:.1f}, {y:.1f})  "
                  f"(image center would be ({img_w / 2:.1f}, {img_h / 2:.1f}))")
            if abs(x - img_w / 2) > 0.05 * img_w or abs(y - img_h / 2) > 0.05 * img_h:
                print("  [WARNING] Center-point prompt is not at the image center.")

    # Prompts alone, drawn on the raw input image
    draw_prompts(arr, boxes, points).save("LN003_frame00000_prompts.png")

    # ---- Predictions ----------------------------------------------------
    pred_instances = result.pred_instances
    masks = pred_instances.masks.cpu().numpy()      # shape (num_instances, H, W), boolean
    scores = pred_instances.scores.cpu().numpy()

    if len(masks) > 0:
        # Save the highest-confidence mask, boolean -> viewable PNG
        best_idx = scores.argmax()
        best_mask = masks[best_idx]

        Image.fromarray(best_mask).save("LN003_frame00000_predft_mask.png")
        print(f"Saved mask with confidence {scores[best_idx]:.3f}")

        overlay = arr.copy()
        overlay[best_mask] = [255, 0, 0]   # tint predicted region red
        blended = (0.6 * arr + 0.4 * overlay).astype(np.uint8)
        Image.fromarray(blended).save("LN003_frame00000_ftoverlay.png")

        # Prompts + predicted mask together on one image
        draw_prompts(blended, boxes, points).save("LN003_frame00000_ftoverlay_prompts.png")
    else:
        print("No predictions above threshold")


if __name__ == "__main__":
    main()
