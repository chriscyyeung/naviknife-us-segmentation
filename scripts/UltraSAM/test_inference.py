import numpy as np
import torch.nn.functional as F
from PIL import Image
from mmdet.apis import init_detector, inference_detector
from endosam.models.utils.custom_functional import multi_head_attention_forward

# Monkey patch the multi_head_attention_forward function
F.multi_head_attention_forward = multi_head_attention_forward


def main():
    config_path = "/home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/external/UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_inference.py"
    ckpt_path = "/home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/external/UltraSam/UltraSam.pth"

    # Initialize the detector
    model = init_detector(config_path, ckpt_path, device='cuda:0')
    print(model.cfg.pretty_text)

    # Load an image
    img = Image.open("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/images/LN003_frame00000.png").convert("RGB")
    arr = np.array(img)
    print(arr.shape)

    # Perform inference
    result = inference_detector(model, arr)

    # Visualize the results
    pred_instances = result.pred_instances
    masks = pred_instances.masks.cpu().numpy()      # shape (num_instances, H, W), boolean
    scores = pred_instances.scores.cpu().numpy()

    if len(masks) > 0:
        # Save the highest-confidence mask, boolean -> viewable PNG
        best_idx = scores.argmax()
        best_mask = masks[best_idx]

        Image.fromarray(best_mask).save("LN003_frame00000_pred_mask.png")
        print(f"Saved mask with confidence {scores[best_idx]:.3f}")

        overlay = arr.copy()
        overlay[best_mask] = [255, 0, 0]   # tint predicted region red
        blended = (0.6 * arr + 0.4 * overlay).astype(np.uint8)
        Image.fromarray(blended).save("LN003_frame00000_overlay.png")
    else:
        print("No predictions above threshold")


if __name__ == "__main__":
    main()
