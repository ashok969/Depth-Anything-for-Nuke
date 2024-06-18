import logging
import cv2
import os
from torch import nn
from torchvision.transforms import Compose
import torch.nn.functional as F
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import torch
from depth_anything.dpt import DepthAnything


logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)
ENCODER = "vitl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = "./nuke/Cattery/DepthAnything/"

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


def main_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14")
        .to(DEVICE)
        .eval()
    )
    return model


class DepthAnythingNuke(nn.Module):
    """DepthAnything model for Nuke."""

    def __init__(self, encoder, decoder) -> None:
        """Initialize the model."""
        super().__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = [20, 21, 22, 23]  # n = 4 on the original code as 'blocks_to_take'

        # Padding
        padding_factor = 14
        pad_h = ((h - 1) // padding_factor + 1) * padding_factor
        pad_w = ((w - 1) // padding_factor + 1) * padding_factor
        pad_dims = (0, pad_w - w, 0, pad_h - h)
        x = torch.nn.functional.pad(x, pad_dims)

        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.encoder.get_intermediate_layers(
            x, n, return_class_token=True
        )

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


def file_size(file_path):
    size_in_bytes = os.path.getsize(file_path)
    return int(size_in_bytes / (1024 * 1024))


if __name__ == "__main__":

    with torch.no_grad():
        depth_anything_model = main_model().eval()

        encoder_model = depth_anything_model.pretrained
        decoder_model = depth_anything_model.depth_head

        model = DepthAnythingNuke(encoder_model, decoder_model)
        model_traced = torch.jit.script(model)
        model_traced = torch.jit.optimize_for_inference(model_traced)

    DESTINATION = f"{BASE_PATH}/DepthAnything_vitl_half.pt"
    model_traced.save(DESTINATION)
    LOGGER.info(
        f"Saved TorchScript model to {DESTINATION} - {file_size(DESTINATION)} MB"
    )

    def test_model(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        depth = depth_anything_model(image)
        depth = depth.detach().cpu().numpy().squeeze()
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype("uint8")
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        cv2.imwrite("depth.png", depth)

    test_model("demo1.png")
