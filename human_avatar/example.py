from pathlib import Path

from PIL import Image

from human_avatar.image_to_avatar import image_to_avatar

example_file = Path(__file__).parent.parent / "assets" / "examples" / "flux" / "source.jpg"

image = Image.open(example_file)
# include_pose primes the holistic models during the Docker build
cropped, masked, pose = image_to_avatar(image, include_pose=True)

masked.show("Masked file")
