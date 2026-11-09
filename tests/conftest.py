import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def background():
    """A textured grayscale image with many corners, larger than the view."""
    rng = np.random.default_rng(0)
    noise = rng.random((720, 960)).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 3.0)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    return (texture * 255).astype(np.uint8)


def warp(image, affine):
    height, width = image.shape
    return cv2.warpAffine(image, affine.astype(np.float64), (width, height), flags=cv2.INTER_LINEAR)
