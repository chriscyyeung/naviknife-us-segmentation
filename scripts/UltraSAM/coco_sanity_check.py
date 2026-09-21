import json

with open("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/coco_annotations.json") as f:
    coco = json.load(f)

print(f"Images: {len(coco['images'])}")
print(f"Annotations: {len(coco['annotations'])}")
print(f"Categories: {coco['categories']}")
print("\nFirst image entry:")
print(json.dumps(coco["images"][0], indent=2))
print("\nFirst annotation entry:")
print(json.dumps(coco["annotations"][0], indent=2))