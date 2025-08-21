from pathlib import Path

from PIL import Image

from human_avatar.image_to_avatar import image_to_avatar

example_file = Path(__file__).parent.parent / "assets" / "examples" / "flux" / "source.jpg"

image = Image.open(example_file)
cropped, masked, pose = image_to_avatar(image)

masked.show("Masked file")
