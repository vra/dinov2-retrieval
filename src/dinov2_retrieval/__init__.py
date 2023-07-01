import argparse

from .image_retriever import ImageRetriver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--model-size",
        type=str,
        default="small",
        choices=["small", "base", "large", "largest"],
        help="DinoV2 model type",
    )
    parser.add_argument(
        "-p",
        "--model-path",
        type=str,
        default=None,
        help="path to dinov2 model, useful when github is unavailable",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=str,
        default="output",
        help="root folder to save output results",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help="path to a query image file or image folder",
    )
    parser.add_argument(
        "-d",
        "--database",
        type=str,
        required=True,
        help="path to the database image file or image folder",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1,
        help="How many images to show in retrieval results",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="image output size",
    )
    parser.add_argument(
        "-m",
        "--margin",
        type=int,
        default=10,
        choices=range(0, 105, 5),
        help="margin size (in pixel) between concatenated images",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="don't cache database features, will extract features each time, quite time-consuming for large database",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show detailed logs",
    )
    args = parser.parse_args()

    image_retriever = ImageRetriver(args)
    image_retriever.run(args)
