# -*- coding: utf-8 -*-
"""
Author: Wei Qin
Date: 2024-12-27
Description:
    Convert each page of a PDF file to an image file and save in the same
    directory of the PDF file.
Update Log:
    2024-12-27: - File created.

"""

import os

import fitz
import numpy as np
from PIL import Image


def pdf_to_images(pdf_path, dpi=200, format='jpg'):
    """
    Convert each page of a PDF file to an image file and save in the same directory of the PDF file.

    Args:
        pdf_path (str): Path to the PDF file
        dpi (int, optional): Resolution of output images. Defaults to 200.
        format (str, optional): Output image format ('jpg', 'png', etc). Defaults to 'jpg'.

    Returns:
        list: List of paths to the generated image files
    """

    # Format mapping
    format_mapping = {
        'jpg': 'JPEG',
        'jpeg': 'JPEG',
        'png': 'PNG',
        'bmp': 'BMP',
        'gif': 'GIF'
    }

    try:
        # Open PDF file
        doc = fitz.open(pdf_path)

        # Set output directory
        output_dir_prefix = os.path.dirname(pdf_path)
        pdf_name = os.path.basename(pdf_path)
        output_dir = os.path.join(
            output_dir_prefix, os.path.splitext(pdf_name)[0] + '_to_images')
        os.makedirs(output_dir, exist_ok=True)

        # Get proper format for PIL
        save_format = format_mapping.get(format.lower(), 'JPEG')

        # List to store image paths
        image_paths = []

        # Convert each page to image
        for page_num in range(len(doc)):
            # Get page
            page = doc[page_num]

            # Set matrix for resolution
            mat = fitz.Matrix(dpi / 72, dpi / 72)

            # Get page pixel map
            pm = page.get_pixmap(matrix=mat, alpha=False)

            # Convert to PIL Image
            img = Image.frombytes('RGB', [pm.width, pm.height], pm.samples)

            # Generate output path
            output_path = os.path.join(
                output_dir,
                f"{pdf_name}_page_{page_num + 1}.{format.lower()}"
            )

            # Save image
            img.save(output_path, format=save_format)
            image_paths.append(output_path)

            print(f"Saved page {page_num + 1} as {output_path}")

        return image_paths

    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")
        return []
    finally:
        if 'doc' in locals():
            doc.close()


if __name__ == "__main__":
    pdf_path = 'path/to/pdf/file.pdf'
    pdf_to_images(
        pdf_path,
        dpi=200,
        format='jpg'
    )
