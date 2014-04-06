"""
Install pip with Freetype2 support

    brew install freetype
    pip install Pillow


The script burns text to image.

Usage:

    python burn-text.py <image> <text>

Example:

    python burn-text.py image.jpg "this text"
"""

import sys

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw


FONT_PATH = "/Library/Fonts/Microsoft/Arial.ttf"


STAMP_COLOR = (255, 255, 255)
STAMP_ALPHA = 255

# Position for timestamp text
# These must be strings, they are evaluated to calculate the final value
X_POSITION = "50"
Y_POSITION = "50"


def draw_stamp(img, stamp):
    width, height = img.size
    font = ImageFont.truetype(FONT_PATH, 120)

    watermark = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(watermark)

    position = (eval(X_POSITION.format(width=width).replace(' ', '')),
                eval(Y_POSITION.format(height=height).replace(' ', '')))
    draw.text(position, stamp, fill=STAMP_COLOR, font=font)

    mask = watermark.convert("L").point(lambda x: min(x, STAMP_ALPHA))
    # Apply this mask to the watermark image, using the alpha filter to
    # make it transparent
    watermark.putalpha(mask)

    # Paste the watermark (with alpha layer) onto the original imag
    img.paste(watermark, None, watermark)


def main():
    image_path = sys.argv[1]
    stamp = ' '.join(sys.argv[2:])

    # Open old image and stamp it with modification time
    img = Image.open(image_path)
    img.convert('RGBA')
    draw_stamp(img, stamp)
    img.save(image_path)


if __name__ == '__main__':
    main()
