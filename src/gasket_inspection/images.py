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


def split_combined_image(
    image: Image.Image, combined_cfg: dict[str, Any]
) -> tuple[Image.Image, Image.Image]:
    layout = combined_cfg.get("layout", "horizontal")
    first_view = combined_cfg.get("first_view", "front")
    ratio = float(combined_cfg.get("split_ratio", 0.5))
    if layout not in {"horizontal", "vertical"}:
        raise ValueError("combined.layout은 horizontal 또는 vertical이어야 합니다.")
    if first_view not in {"front", "side"}:
        raise ValueError("combined.first_view는 front 또는 side여야 합니다.")
    if not 0.05 < ratio < 0.95:
        raise ValueError("combined.split_ratio는 0.05보다 크고 0.95보다 작아야 합니다.")

    width, height = image.size
    if layout == "horizontal":
        boundary = round(width * ratio)
        first = image.crop((0, 0, boundary, height))
        second = image.crop((boundary, 0, width, height))
    else:
        boundary = round(height * ratio)
        first = image.crop((0, 0, width, boundary))
        second = image.crop((0, boundary, width, height))

    if first.width == 0 or first.height == 0 or second.width == 0 or second.height == 0:
        raise ValueError("합성 이미지 분할 결과가 비어 있습니다. split_ratio를 확인하세요.")
    return (first, second) if first_view == "front" else (second, first)


def load_view_pair(
    input_cfg: dict[str, Any],
    *,
    front_path: str | Path | None = None,
    side_path: str | Path | None = None,
    combined_path: str | Path | None = None,
) -> tuple[Image.Image, Image.Image]:
    mode = input_cfg["mode"]
    if mode == "paired_files":
        if front_path is None or side_path is None:
            raise ValueError("paired_files 모드에는 front_path와 side_path가 모두 필요합니다.")
        return open_rgb(front_path), open_rgb(side_path)
    if combined_path is None:
        raise ValueError("combined_image 모드에는 combined_path가 필요합니다.")
    combined = open_rgb(combined_path)
    return split_combined_image(combined, input_cfg["combined"])


def load_view_pair_bytes(
    input_cfg: dict[str, Any],
    *,
    front_bytes: bytes | None = None,
    side_bytes: bytes | None = None,
    combined_bytes: bytes | None = None,
) -> tuple[Image.Image, Image.Image]:
    mode = input_cfg["mode"]
    if mode == "paired_files":
        if front_bytes is None or side_bytes is None:
            raise ValueError("paired_files 모드에는 front/side bytes가 모두 필요합니다.")
        return (
            open_rgb_bytes(front_bytes, "front image bytes"),
            open_rgb_bytes(side_bytes, "side image bytes"),
        )
    if combined_bytes is None:
        raise ValueError("combined_image 모드에는 combined bytes가 필요합니다.")
    combined = open_rgb_bytes(combined_bytes, "combined image bytes")
    return split_combined_image(combined, input_cfg["combined"])


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
