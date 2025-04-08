from pdf2image import convert_from_path
from PIL import Image

pdf_files = [f"data/img/gamestate{i}.pdf" for i in range(5,10)]
# Convert PDF pages to images at 300 dpi
for pdf_path in pdf_files:
    pages = convert_from_path(pdf_path, dpi=600)

    # Determine the width (max width of any page) and total height (sum of pages) for vertical stacking
    width = max(page.width for page in pages)
    height = sum(page.height for page in pages)

    # Create a new blank image with the calculated size
    combined_image = Image.new('RGB', (width, height))

    # Paste each page into the combined image sequentially (vertically)
    y_offset = 0
    for page in pages:
        combined_image.paste(page, (0, y_offset))
        y_offset += page.height

    # Save the result
    combined_image.save(pdf_path.replace("pdf", "png"))
