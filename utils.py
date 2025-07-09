from PIL import Image
import torch

def combine_images_horizontally(images):
    """
    Combine a list of PIL.Image objects into a single image horizontally.

    :param images: List of PIL.Image objects
    :return: A single PIL.Image object
    """
    # Calculate total width and maximum height
    total_width = sum(image.width for image in images)
    max_height = max(image.height for image in images)

    # Create a new blank image with the calculated dimensions
    combined_image = Image.new("RGB", (total_width, max_height))

    # Paste each image into the new image
    x_offset = 0
    for image in images:
        combined_image.paste(image, (x_offset, 0))
        x_offset += image.width

    return combined_image

def combine_images_vertically(images):
    """
    Combine a list of PIL.Image objects into a single image vertically.

    :param images: List of PIL.Image objects
    :return: A single PIL.Image object
    """
    # Calculate total height and maximum width
    total_height = sum(image.height for image in images)
    max_width = max(image.width for image in images)

    # Create a new blank image with the calculated dimensions
    combined_image = Image.new("RGB", (max_width, total_height))

    # Paste each image into the new image
    y_offset = 0
    for image in images:
        combined_image.paste(image, (0, y_offset))
        y_offset += image.height

    return combined_image

def is_in_range(tensor, min_value, max_value):
    return torch.min(tensor) >= min_value and torch.max(tensor) <= max_value