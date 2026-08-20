from __future__ import annotations

from PIL import Image

from gasket_inspection.images import SquarePad


def test_square_pad_preserves_entire_image() -> None:
    image = Image.new("RGB", (100, 40), "white")
    padded = SquarePad((0, 0, 0))(image)
    assert padded.size == (100, 100)
    assert padded.getpixel((50, 50)) == (255, 255, 255)
    assert padded.getpixel((50, 0)) == (0, 0, 0)
