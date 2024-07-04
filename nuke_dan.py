"""Export the DepthAnything model for Nuke."""

import argparse
import logging
import os
from typing import List

import cv2
import torch
import torch.nn.functional as F
from packaging import version
from torch import nn
from torchvision.transforms import Compose
from v1.dpt import DPT_DINOv2
from v1.util.transform import NormalizeImage, PrepareForNet, Resize
from v2.dpt import DepthAnythingV2

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = "./nuke/Cattery/DepthAnything"
IS_TORCH_1_12 = version.parse(torch.__version__) >= version.parse("1.12.0")
MODEL_CONFIG = {
    "v1_vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "url": "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vits14.pth",
    },
    "v1_vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "url": "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitb14.pth",
    },
    "v1_vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "url": "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth",
    },
    "v2_vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    },
    "v2_vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
    },
    "v2_vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
    },
    "v2_vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Giant/resolve/main/depth_anything_v2_vitg.pth",
    },  # temporarily offline
}

transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)


def build_depth_anything_model(version: str, model_size: str, use_half=False):
    """Load the DepthAnythingV1 model.

    Models will be downloaded from the Hugging Face Hub
    https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main

    Args:
        version: The DepthAnything version (v1, v2).
        model_size: The depth_anything model_size (vits, vitb, vitl).
        use_half: Whether to use half precision.

    Returns
        torch.nn.Module: The DepthAnythingV1 model
    """
    model_key = f"{version}_{model_size}"
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Invalid version or model size: {model_key}")

    model_config = MODEL_CONFIG[model_key].copy()
    url = model_config.pop("url")
    model_file = os.path.basename(url)

    if not os.path.exists(model_file):
        LOGGER.info("Downloading model file: %s", model_file)
        torch.hub.download_url_to_file(url, model_file)

    if version == "v1":
        model = DPT_DINOv2(**model_config, use_bn=False, use_clstoken=False)
    elif version == "v2":
        model = DepthAnythingV2(**model_config)

    state_dict = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state_dict)

    if use_half:
        model = model.half()
    return model.to(DEVICE).eval()


def file_size(file_path):
    """Get the file size in MB."""
    size_in_bytes = os.path.getsize(file_path)
    return int(size_in_bytes / (1024 * 1024))


def test_model(image_path, model, version: str, model_size: str, use_half=False):
    """Test the model with an image.

    Args:
        image_path: The path to the image.
        model: The model to test.
        version: The DepthAnything version (v1, v2).
        model_size: The depth_anything model_size (vits, vitb, vitl).
        use_half: Whether to use half precision.
    """
    LOGGER.info("Testing model with image: %s", image_path)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image file: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({"image": image})["image"]
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    if use_half:
        image = image.half()

    with torch.no_grad():
        depth = model(image)

    # Normalize and convert to uint8 for visualization
    depth = depth.detach().cpu().numpy().squeeze()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype("uint8")
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

    dest = f"depth_{version}_{model_size}_{'fp16' if use_half else 'fp32'}_{os.urandom(2).hex()}.png"
    cv2.imwrite(dest, depth)
    LOGGER.info("Test image saved to %s", dest)


class DepthAnythingNuke(nn.Module):
    """DepthAnything model for Nuke.

    Args:
        encoder: The encoder model.
        decoder: The decoder model.
        n: Depth Anything window list parameter.
    """

    def __init__(self, encoder, decoder, n: List[int], use_half=False) -> None:
        """Initialize the model."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n = n
        self.use_half = use_half

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.n
        b, c, h, w = x.shape
        device = torch.device("cuda") if x.is_cuda else torch.device("cpu")

        # Padding
        padding_factor = 14
        pad_h = ((h - 1) // padding_factor + 1) * padding_factor
        pad_w = ((w - 1) // padding_factor + 1) * padding_factor
        pad_dims = (0, pad_w - w, 0, pad_h - h)
        x = F.pad(x, pad_dims)

        std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=device).view(
            1, 3, 1, 1
        )
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=device).view(
            1, 3, 1, 1
        )
        x = (x - mean) / std

        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.encoder.get_intermediate_layers(x, n, return_class_token=True)
        depth = self.decoder(features, patch_h, patch_w)
        depth = F.relu(depth)
        depth = depth.squeeze(1)
        depth = F.interpolate(
            depth[:, None],
            size=(x.shape[-2], x.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        return depth[:, :, :h, :w]


def trace_depth_anything(version: str, model_size: str, use_half=False):
    """Trace the DepthAnythingV2 model.

    Returns
        torch.jit.ScriptModule: The traced model.
    """
    LOGGER.info(
        "Tracing DepthAnything: %s_%s %s", version, model_size, "(half)" if use_half else ""
    )

    depth_anything_model = build_depth_anything_model(version, model_size, use_half)

    model = DepthAnythingNuke(
        encoder=depth_anything_model.pretrained,
        decoder=depth_anything_model.depth_head,
        n=depth_anything_model.n,
        use_half=use_half,
    )
    model = model.half()
    model_traced = torch.jit.script(model)

    if IS_TORCH_1_12:
        model_traced = torch.jit.optimize_for_inference(model_traced)

    DEST = (
        f"{BASE_PATH}/DepthAnything_{version}_{model_size}{'_half' if use_half else ''}.pt"
    )
    model_traced.save(DEST)
    LOGGER.info("TorchScript model saved to %s (%sMB)", DEST, file_size(DEST))
    return model_traced


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export the DepthAnything model for Nuke"
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["v1", "v2"],
        default="v2",
        help="DepthAnything version",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["vits", "vitb", "vitl"],
        default="vits",
        help="Model size",
    )
    parser.add_argument(
        "--half", action="store_true", default=False, help="Use half precision"
    )
    parser.add_argument(
        "--test-image", type=str, default="demo.png", help="Path to test image"
    )

    args = parser.parse_args()

    # model = depth_anything_model(args.version, args.model_size, args.half)
    model = trace_depth_anything(args.version, args.model_size, args.half)
    # test_model(args.test_image, model, args.version, args.model_size, args.half)
