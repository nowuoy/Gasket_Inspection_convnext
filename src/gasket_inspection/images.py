from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class SquarePad:
    """원본 전체를 보존한 채 정사각형으로 padding합니다."""

    def __init__(self, fill: tuple[int, int, int]) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=self.fill)


def open_rgb(path: str | Path) -> Image.Image:
    path = Path(path)
    try:
        with Image.open(path) as image:
            image.load()  # 잘린 파일/부분 저장 파일을 여기서 검출합니다.
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError(f"이미지를 해석할 수 없습니다: {path}") from exc


def open_rgb_bytes(data: bytes, source_name: str) -> Image.Image:
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError(f"이미지를 해석할 수 없습니다: {source_name}") from exc


def build_transform(input_cfg: dict[str, Any], train_cfg: dict[str, Any] | None = None):
    fill = tuple(int(value) for value in input_cfg.get("padding_value", (0, 0, 0)))
    image_size = int(input_cfg["image_size"])
    operations: list[Any] = [SquarePad(fill)]

    if train_cfg is not None:
        aug = train_cfg.get("augmentation", {})
        rotation = float(aug.get("rotation_degrees", 0.0))
        translate = float(aug.get("translate_fraction", 0.0))
        if rotation > 0 or translate > 0:
            operations.append(
                transforms.RandomAffine(
                    degrees=rotation,
                    translate=(translate, translate),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=fill,
                )
            )
        brightness = float(aug.get("brightness", 0.0))
        contrast = float(aug.get("contrast", 0.0))
        saturation = float(aug.get("saturation", 0.0))
        if any(value > 0 for value in (brightness, contrast, saturation)):
            operations.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=0.0,
                )
            )

    operations.extend(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=input_cfg["mean"], std=input_cfg["std"]),
        ]
    )
    return transforms.Compose(operations)
