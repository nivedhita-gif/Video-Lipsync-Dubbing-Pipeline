"""
musetalk_patches.py

MuseTalk's cloned repo requires one small patch to run correctly with
recent versions of PyTorch (2.6+), which changed the default behavior
of torch.load() to be more restrictive about loading old model files.

This file documents and applies that single patch programmatically,
so the same fix can be applied identically on the laptop or on Colab.
"""

import os


def patch_resnet_weights_only(musetalk_dir: str):
    """
    Fixes musetalk/utils/face_parsing/resnet.py

    Original line:
        state_dict = torch.load(model_path)

    Patched line:
        state_dict = torch.load(model_path, weights_only=False)

    Why: PyTorch 2.6 changed torch.load()'s default `weights_only`
    from False to True. The face-parsing ResNet18 checkpoint is an
    older-format file that fails to load under the new strict default.
    Since this checkpoint comes from a trusted source (downloaded
    directly from PyTorch's model zoo / HuggingFace), it's safe to
    set weights_only=False explicitly.
    """
    path = os.path.join(musetalk_dir, "musetalk", "utils", "face_parsing", "resnet.py")

    with open(path, "r") as f:
        content = f.read()

    old = "state_dict = torch.load(model_path)"
    new = "state_dict = torch.load(model_path, weights_only=False)"

    if new in content:
        print("Patch already applied, skipping.")
        return

    if old not in content:
        print("WARNING: expected line not found, patch not applied.")
        return

    content = content.replace(old, new)

    with open(path, "w") as f:
        f.write(content)

    print(f"Patched: {path}")


def apply_all_patches(musetalk_dir: str):
    """Applies every known patch to a MuseTalk clone."""
    patch_resnet_weights_only(musetalk_dir)


if __name__ == "__main__":
    # When run directly, patches the local MuseTalk clone
    MUSETALK_DIR = r"C:\lipsync\MuseTalk"
    apply_all_patches(MUSETALK_DIR)