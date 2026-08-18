from __future__ import annotations

from PIL import Image

from gasket_inspection.images import split_combined_image


def test_horizontal_combined_image_front_first() -> None:
    image = Image.new("RGB", (100, 40))
    front, side = split_combined_image(
        image,
        {"layout": "horizontal", "first_view": "front", "split_ratio": 0.4},
    )
    assert front.size == (40, 40)
    assert side.size == (60, 40)


def test_vertical_combined_image_side_first() -> None:
    image = Image.new("RGB", (30, 100))
    front, side = split_combined_image(
        image,
        {"layout": "vertical", "first_view": "side", "split_ratio": 0.3},
    )
    assert front.size == (30, 70)
    assert side.size == (30, 30)

