#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Generation Script
-----------------------

This script prepares the images required for isolated pixel detection experiments.
It loads real images, generates synthetic ones, applies optional cropping or
smoothing, and inserts artificial isolated pixels when needed.

All detection is performed by a separate script. This file is *only* responsible
for producing the input images used by those experiments.
"""
# %% -------- Imports --------
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from noise import snoise2
import matplotlib.pyplot as plt

# %% ===================================================================
# Paths
# ======================================================================

def data_path():
    """Returns path to the data directory two levels above this file."""
    return Path(__file__).resolve().parents[2] / "data"


def save_image(np_img, filename):
    # Convert any dtype to uint8 for safe saving
    if np_img.dtype != np.uint8:
        arr8 = (np_img / np_img.max() * 255).astype(np.uint8)
        np_img = arr8

    cv2.imwrite(str(data_path() / filename), np_img)


def show_image(np_img):

    img = Image.fromarray(np_img)
    
    # Convert 16-bit grayscale ("I") to 8-bit ("L")
    if img.mode == "I":
        arr = np.array(img, dtype=np.uint16)
        arr8 = (arr / 256).astype(np.uint8)  # exact mapping 0–65535 → 0–255
        img8 = Image.fromarray(arr8, mode="L")
    else:
        img8 = img.convert("L")

    # Show properly
    plt.figure(figsize=(5,5))
    plt.imshow(img8, cmap="gray")
    plt.axis("off")
    plt.show()


# %% ===================================================================
# Utility Functions
# ======================================================================

def generate_perlin_noise(width, height, scale=300.0, octaves=6,
                          persistence=0.5, lacunarity=3.0, base=42):
    """Generate Perlin-noise greyscale image."""
    xs = np.linspace(0, width / scale, width)
    ys = np.linspace(0, height / scale, height)

    noise_data = np.array([
        [snoise2(x, y,
                 octaves=octaves,
                 persistence=persistence,
                 lacunarity=lacunarity,
                 repeatx=1024, repeaty=1024,
                 base=base)
         for x in xs]
        for y in ys
    ])

    norm = (noise_data - noise_data.min()) / (noise_data.max() - noise_data.min())
    return (norm * 255).astype(np.uint8)


def add_isolated_pixels(img_array, n=10, delta_range=(20, 60)):
    """Randomly modify n single pixels to create isolated anomalies."""
    h, w = img_array.shape
    out = img_array.copy()
    for _ in range(n):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        delta = np.random.uniform(*delta_range)
        v = out[y, x]
        out[y, x] = np.clip(v + delta if np.random.rand() > 0.5 else v - delta, 0, 255)
    return out


# %% ===================================================================
# Image construction functions
# ======================================================================

def build_ab1_cropped():
    """Load ab1.png, convert to greyscale, crop region, save result."""
    src = data_path() / "ab1.png"
    img = Image.open(src)
    np_img = np.array(img)
    cropped_arr = np_img[200:600, 400:800]
    corrupted = add_isolated_pixels(cropped_arr, n=100)
    save_image(corrupted, "ab1_cropped.png")
    return corrupted


def build_spatial_anomaly_nodes():
    """Create Perlin-noise image, smooth, add isolated pixels."""
    width, height = 128, 128
    base_noise = generate_perlin_noise(width, height)
    smoothed = np.array(Image.fromarray(base_noise).filter(ImageFilter.GaussianBlur(radius=2)))
    corrupted = add_isolated_pixels(smoothed, n=10)
    save_image(corrupted, "spatial_anomaly_nodes.png")
    return corrupted


# ======================================================================
# Registry of buildable images
# ======================================================================

IMAGE_BUILDERS = {
    "ab1_cropped.png": build_ab1_cropped,
    "spatial_anomaly_nodes.png": build_spatial_anomaly_nodes,
}


# ======================================================================
# Main entry
# ======================================================================

def build_images(requested=None, show=False):
    """
    Build one or all images.
    requested: name of a specific file, or None to build everything.
    """
    if requested is None:
        for name, fn in IMAGE_BUILDERS.items():
            print(f"Generating {name} ...")
            img = fn()
            if show:
                show_image(img)
        print("All images generated.")
        return

    # Build only selected
    if requested not in IMAGE_BUILDERS:
        raise ValueError(f"Unknown image: {requested}. "
                         f"Available: {list(IMAGE_BUILDERS.keys())}")

    print(f"Generating {requested} ...")
    img = IMAGE_BUILDERS[requested]()
    if show:
        show_image(img)

# %%
build_images(requested="ab1_cropped.png", show=True)
# %%
