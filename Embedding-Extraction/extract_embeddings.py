import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
VAL_FRACTION = 0.10
SPLIT_SEED = 12345
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
DEFAULT_BATCH = {"resnet50": 256, "phikon": 256, "uni": 256, "hoptimus0": 128}


def list_images(root):
    paths, labels = [], []
    for c_idx, c in enumerate(CLASSES):
        for f in sorted(os.listdir(os.path.join(root, c))):
            if f.lower().endswith(IMG_EXT):
                paths.append(f"{c}/{f}")
                labels.append(c_idx)
    return np.array(paths), np.array(labels)


def tensor_collate(tf):
    return lambda imgs: torch.stack([tf(im) for im in imgs])


def load_encoder(name, hf_token):
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    if name == "resnet50":
        import torchvision
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return model, tensor_collate(tf), lambda m, x: m(x)

    if name == "phikon":
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained("owkin/phikon")
        model = AutoModel.from_pretrained("owkin/phikon")
        collate = lambda imgs: proc(images=imgs, return_tensors="pt")["pixel_values"]
        return model, collate, lambda m, x: m(pixel_values=x).last_hidden_state[:, 0]

    import timm
    if name == "uni":
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True,
                                  init_values=1e-5, dynamic_img_size=True)
        tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                 transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return model, tensor_collate(tf), lambda m, x: m(x)

    if name == "hoptimus0":
        model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True,
                                  init_values=1e-5, dynamic_img_size=False)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(HOPTIMUS_MEAN, HOPTIMUS_STD)])
        return model, tensor_collate(tf), lambda m, x: m(x)

    raise ValueError(f"unknown encoder {name}")


class ImageSet(torch.utils.data.Dataset):
    def __init__(self, root, paths):
        self.root, self.paths = root, paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return Image.open(os.path.join(self.root, self.paths[i])).convert("RGB")


@torch.no_grad()
def extract(model, collate, embed, root, paths, batch_size, num_workers, device):
    loader = torch.utils.data.DataLoader(ImageSet(root, paths), batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, collate_fn=collate, pin_memory=True)
    feats = []
    for x in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = embed(model, x.to(device, non_blocking=True))
        feats.append(out.float().cpu())
    return torch.cat(feats).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--encoder", required=True, choices=["resnet50", "phikon", "uni", "hoptimus0"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or DEFAULT_BATCH[args.encoder]

    train_paths, train_labels = list_images(args.train_root)
    test_paths, test_labels = list_images(args.test_root)
    tr_idx, va_idx = train_test_split(np.arange(len(train_paths)), test_size=VAL_FRACTION,
                                      stratify=train_labels, random_state=SPLIT_SEED)
    tr_idx, va_idx = np.sort(tr_idx), np.sort(va_idx)
    splits = {
        "train": (args.train_root, train_paths[tr_idx], train_labels[tr_idx]),
        "val": (args.train_root, train_paths[va_idx], train_labels[va_idx]),
        "test": (args.test_root, test_paths, test_labels),
    }

    model, collate, embed = load_encoder(args.encoder, args.hf_token)
    model = model.to(device).eval()

    meta = {"encoder": args.encoder, "split_seed": SPLIT_SEED, "val_fraction": VAL_FRACTION, "splits": {}}
    for split, (root, paths, labels) in splits.items():
        X = extract(model, collate, embed, root, list(paths), batch_size, args.num_workers, device)
        np.save(os.path.join(args.out_dir, f"feats_{split}.npy"), X.astype(np.float32))
        np.save(os.path.join(args.out_dir, f"labels_{split}.npy"), labels.astype(np.int64))
        np.save(os.path.join(args.out_dir, f"ids_{split}.npy"), paths.astype(str))
        meta["embedding_dim"] = int(X.shape[1])
        meta["splits"][split] = {"n": int(len(paths)),
                                 "class_counts": {c: int((labels == i).sum()) for i, c in enumerate(CLASSES)}}
        print(f"{split}: {X.shape}", flush=True)

    with open(os.path.join(args.out_dir, "extract_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
