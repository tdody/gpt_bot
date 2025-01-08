"""
TINY_WRITR is a module that allows the user to find more inspiration from a list of content.
For instance, you want to create a new movie title, a new book title, add a new character to your universe,
or even name a new bird. TINY_WRITR will help you generate a list of content that you can use to inspire you.


"""

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tiny Writr is a module that allows the user to find more inspiration from a list of content."
    )

    parser.add_argument(
        "--input_file", "-i", type=str, help="Path to the input file.", required=True
    )
    parser.add_argument(
        "--workings_dir",
        "-w",
        type=str,
        help="The directory where the files will be saved.",
        default="output.txt",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model to use for generating content.",
        default="bigram",
    )
