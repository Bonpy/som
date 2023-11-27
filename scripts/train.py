import argparse
import os

import numpy as np

from colour_som import SOMFast


def parse_args():
    """Parses CLI arguments"""
    parser = argparse.ArgumentParser(description="Train SOM for colour categorisation")

    parser.add_argument(
        "--colours",
        type=int,
        default=int(os.getenv("COLOURS", 10)),
        help="Number of colours to categorise",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.getenv("ITERATIONS", 100)),
        help="Number of iterations for training",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=os.getenv("OUTPUT", "output.png"),
        help="Output image file name",
    )

    parser.add_argument(
        "--grid-width",
        type=int,
        default=int(os.getenv("GRID_WIDTH", 100)),
        help="Grid width of the SOM",
    )
    parser.add_argument(
        "--grid-height",
        type=int,
        default=int(os.getenv("GRID_HEIGHT", 100)),
        help="Grid height of the SOM",
    )

    parser.add_argument(
        "--input-dim",
        type=int,
        default=int(os.getenv("INPUT_DIM", 3)),
        help="Dimensionality of the input space",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=float(os.getenv("LR", 0.1)),
        help="Initial learning rate",
    )

    return parser.parse_args()


def main():
    """Entrypoint for this script."""
    args = parse_args()
    print(
        "Training with the following parameters:\n"
        f"  Colours: {args.colours}\n"
        f"  Iterations: {args.iterations}\n"
        f"  Output: {args.output}\n"
        f"  Grid Width: {args.grid_width}\n"
        f"  Grid Height: {args.grid_height}\n"
        f"  Input Dimension: {args.input_dim}\n"
        f"  Learning Rate: {args.lr}"
    )

    input_data = np.random.random((args.colours, args.input_dim))
    som = SOMFast((args.grid_width, args.grid_height), args.input_dim)
    som.train(input_data, args.iterations)
    som.visualise(args.output)


if __name__ == "__main__":
    main()
