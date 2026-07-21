import random
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw


# ===================== Configuration =====================
# /path/to/ImageNet/train/
BASE_PATH = Path("/home/x_yiflu/dataset/Datasets/imagenet256-StyleGAN-XL")    # /home/x_yiflu/dataset/Datasets/imagenet256/train # /home/x_yiflu/dataset/Datasets/Imagenet256-BigGAN #/home/x_yiflu/dataset/Datasets/Imagenet256-LDM

# Directory for saving the 3x3 image grids
OUTPUT_PATH = Path("./stylegan_grids")

# Number of folders (classes) to randomly select
NUM_FOLDERS = 20

# Number of images to sample from each folder
NUM_IMAGES_PER_GRID = 9

# Size of each image in the grid
IMAGE_SIZE = (256, 256)

# Supported image file extensions
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

# Set a random seed for reproducibility
# Remove this line if reproducibility is not needed
random.seed(42)
# =========================================================


def load_and_resize_image(image_path: Path, image_size: tuple[int, int]) -> Image.Image:
    """
    Load an image, preserve its aspect ratio, and center-crop it to the target size.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        # Resize while preserving aspect ratio, then center crop
        image = ImageOps.fit(
            image,
            image_size,
            method=Image.Resampling.LANCZOS
        )

        return image.copy()


def create_grid(
    image_paths: list[Path],
    output_file: Path,
    image_size: tuple[int, int] = (256, 256),
    rows: int = 3,
    cols: int = 3,
    gap: int = 4,
) -> None:
    """
    Create an image grid with the specified number of rows and columns.
    """
    cell_width, cell_height = image_size

    grid_width = cols * cell_width + (cols - 1) * gap
    grid_height = rows * cell_height + (rows - 1) * gap

    grid = Image.new(
        "RGB",
        (grid_width, grid_height),
        color="white"
    )

    for index, image_path in enumerate(image_paths):
        row = index // cols
        col = index % cols

        x = col * (cell_width + gap)
        y = row * (cell_height + gap)

        try:
            image = load_and_resize_image(image_path, image_size)
            grid.paste(image, (x, y))

        except Exception as error:
            print(f"Failed to load image {image_path}: {error}")

            # Use a placeholder image if loading fails
            placeholder = Image.new("RGB", image_size, color="lightgray")
            draw = ImageDraw.Draw(placeholder)
            draw.text((10, 10), "Invalid image", fill="black")
            grid.paste(placeholder, (x, y))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_file, quality=95)

    print(f"Saved image grid to: {output_file}")


def find_image_files(folder_path: Path) -> list[Path]:
    """
    Return all supported image files in the given folder.
    """
    return sorted(
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Randomly select 20 unique numbers between 1 and 1000
    # random_numbers = random.sample(range(0, 999), NUM_FOLDERS)

    random_numbers = [654, 114, 25, 759, 281, 250, 228, 142, 754, 104, 692, 758, 913, 558, 89, 604, 432, 32, 30, 95]

    print("Randomly selected folder IDs:")
    print(random_numbers)

    successful_numbers = []

    for number in random_numbers:
        # Construct the folder path using the selected number
        folder_path = BASE_PATH / str(number)

        if not folder_path.exists():
            print(f"Folder does not exist, skipping: {folder_path}")
            continue

        if not folder_path.is_dir():
            print(f"Path is not a directory, skipping: {folder_path}")
            continue

        image_files = find_image_files(folder_path)

        if len(image_files) < NUM_IMAGES_PER_GRID:
            print(
                f"Folder {folder_path} contains only {len(image_files)} images "
                f"(requires at least {NUM_IMAGES_PER_GRID}), skipping."
            )
            continue

        # Randomly sample 9 images without replacement
        selected_images = random.sample(
            image_files,
            NUM_IMAGES_PER_GRID
        )

        output_file = OUTPUT_PATH / f"grid_{number}.png"

        create_grid(
            image_paths=selected_images,
            output_file=output_file,
            image_size=IMAGE_SIZE,
            rows=3,
            cols=3,
        )

        successful_numbers.append(number)

    print("\nFinished.")
    print(f"Successfully created {len(successful_numbers)} image grids.")
    print(f"Processed folder IDs: {successful_numbers}")


if __name__ == "__main__":
    main()