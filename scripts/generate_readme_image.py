"""A script to generate the image for README.md."""

import pathlib

import fbench


def main():
    image_path = pathlib.Path().cwd() / "images" / "readme-ackley.png"
    plotter = fbench.viz.FunctionPlotter(func=fbench.ackley, bounds=[(-5, 5)] * 2)
    fig, _, _ = plotter.plot()
    fig.savefig(image_path)
    print(f"saved {image_path}")


if __name__ == "__main__":
    main()
