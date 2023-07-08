from collections import namedtuple
import os
from pathlib import Path
import sys

import pytest

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, src_dir)
from dinov2_retrieval.image_retriever import ImageRetriver


def test_init():
    args = {
        "verbose": True,
        "model_size": "small",
        "model_path": None,
        "num": 5,
        "disable_cache": False,
        "database": "/path/to/database",
    }
    Args = namedtuple("Args", list(args.keys()))
    args = Args(**args)
    image_retriver = ImageRetriver(args)
    assert image_retriver.top_k == 5
    assert image_retriver.model_name == "dinov2_vits14"


def test_glob_images(tmpdir):
    # create some test image files
    tmpdir.join("test1.jpg").write("test")
    tmpdir.join("test2.png").write("test")
    tmpdir.join("test3.bmp").write("test")

    args = {
        "verbose": True,
        "model_size": "small",
        "model_path": None,
        "num": 5,
        "disable_cache": False,
        "database": "/path/to/database",
    }
    Args = namedtuple("Args", list(args.keys()))
    args = Args(**args)
    image_retriver = ImageRetriver(args)

    tmpdir = Path(tmpdir)
    images = image_retriver.glob_images(tmpdir)
    assert len(images) == 3
    assert set(images) == set(
        [tmpdir / "test1.jpg", tmpdir / "test2.png", tmpdir / "test3.bmp"]
    )


# vim: ts=4 sw=4 sts=4 expandtab
