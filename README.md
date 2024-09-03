# Image to Human Avatar

This project aims to automatically process photos of humans into animatable human avatars.

Specifically, we aim to detect the human in the image, position it to our standard positioning, and mask it.

|                   Source Image                    |                            Area of Interest                            |                  Cropped                  |                  Masked                  |
|:-------------------------------------------------:|:----------------------------------------------------------------------:|:-----------------------------------------:|:----------------------------------------:|
| High quality image of a human. Shoulder width $x$ | Area of interest in $2.5x$ shoulder width where the center is the neck |           Image after cropping            |           Image after masking            |
|     ![source](assets/explanation/source.png)      |                 ![source](assets/explanation/aoi.png)                  | ![source](assets/explanation/cropped.png) | ![source](assets/explanation/masked.png) |

## Real Examples

|                Source Image                 |                   Avatar                    |                Masked Avatar                |
|:-------------------------------------------:|:-------------------------------------------:|:-------------------------------------------:|
| ![source](assets/examples/stock/source.jpg) | ![avatar](assets/examples/stock/avatar.jpg) | ![masked](assets/examples/stock/masked.png) |
| ![source](assets/examples/amit/source.jpg)  | ![avatar](assets/examples/amit/avatar.jpg)  | ![masked](assets/examples/amit/masked.png)  |

## Conditions

- Input can be of any aspect ratio.
- There must be a human in the image (real or fictional).
- The human must be standing, with their arms down, facing the camera (TODO).
- The image must be safe-for-work.
- The creator of the avatar must have the right to use the image.

## Usage

```bash
pip install git+https://github.com/sign/image-to-human-avatar
```

To then process an image:
```python
from human_avatar import image_to_avatar
from PIL import Image

image = Image.open("example.jpg")
cropped, masked, pose = image_to_avatar(image)

masked.save("masked.png")
```