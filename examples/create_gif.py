import argparse
from pathlib import Path
from PIL import Image
import re

def create_gif(input_folder: Path, output_file: Path, duration: int, loop: int):
    """
    Creates an animated GIF from all PNG files in a specified folder.

    Args:
        input_folder: The path to the folder containing PNG images.
        output_file: The path (including filename) for the output GIF.
        duration: The duration (in milliseconds) for each frame in the GIF.
        loop: The number of times the GIF should loop (0 means loop forever).
    """
    if not input_folder.is_dir():
        print(f"Error: Input folder not found at '{input_folder}'")
        return

    # Find all .png files in the input folder
    image_paths = list(input_folder.glob('*.png'))

    if not image_paths:
        print(f"Error: No .png files found in '{input_folder}'")
        return

    # --- Smart Sorting ---
    # Sort files naturally based on numbers in the filename.
    # This correctly handles 'ep1', 'ep2', 'ep10' etc.
    def natural_sort_key(filename):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', str(filename))]

    image_paths.sort(key=natural_sort_key)

    print(f"Found {len(image_paths)} images. Creating GIF...")
    print("Frame order:")
    for i, path in enumerate(image_paths):
        print(f"  {i+1}: {path.name}")

    # Open all images and store them in a list
    try:
        frames = [Image.open(path) for path in image_paths]
    except Exception as e:
        print(f"Error opening images: {e}")
        return

    # The first frame is the starting point
    first_frame = frames[0]

    # Save the first frame, and append the rest
    first_frame.save(
        output_file,
        format='GIF',
        append_images=frames[1:], # Append all frames after the first
        save_all=True,
        duration=duration,       # Time per frame in ms
        loop=loop                # 0 for infinite loop
    )

    print(f"\nSuccessfully created GIF: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create an animated GIF from a folder of PNG images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input-folder',
        type=Path,
        required=True,
        help="The folder containing the PNG images to compile."
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default='animation.gif',
        help="The name of the output GIF file."
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=200,
        help="Duration of each frame in the GIF, in milliseconds."
    )
    parser.add_argument(
        '--loop',
        type=int,
        default=0,
        help="Number of times the GIF should loop. Use 0 for an infinite loop."
    )

    args = parser.parse_args()

    create_gif(args.input_folder, args.output_file, args.duration, args.loop)
