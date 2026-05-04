from functools import cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pose_format import Pose
from pose_format.utils.generic import pose_normalization_info
from pose_format.utils.holistic import load_holistic
from torchvision import transforms
from transformers import AutoModelForImageSegmentation, pipeline

CROP_RESOLUTION = 512
RMBG_INPUT_SIZE = (1024, 1024)


def extract_pose(image: Image):
    frames = [np.array(image)]
    pose = load_holistic(frames,
                         fps=1,
                         width=image.width,
                         height=image.height,
                         depth=image.width,
                         additional_holistic_config={
                             "model_complexity": 2,
                             "smooth_landmarks": False,
                             "refine_face_landmarks": True,
                         })
    if pose.body.data.mask.all():
        raise ValueError("No pose detected")
    return pose


@cache
def load_huggingface_model(task: str, model: str):
    return pipeline(task=task, model=model, trust_remote_code=True)


def is_safe_for_work(image: Image):
    model = load_huggingface_model("image-classification", model="Falconsai/nsfw_image_detection")
    return model(image)[0]["label"] == "normal"


def crop_person(image: Image, pose: Pose):
    normalization_info = pose_normalization_info(pose.header)

    l_shoulder = pose.body.data[0, 0, normalization_info.p1]
    r_shoulder = pose.body.data[0, 0, normalization_info.p2]

    center_x = abs((l_shoulder[0] + r_shoulder[0]) / 2)
    center_y = abs((l_shoulder[1] + r_shoulder[1]) / 2)
    crop_size = 1.25 * abs(l_shoulder[0] - r_shoulder[0])

    image = image.crop((int(center_x - crop_size),
                        int(center_y - crop_size),
                        int(center_x + crop_size),
                        int(center_y + crop_size)))

    return image


@cache
def load_rmbg_model():
    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, device


def remove_image_background(image: Image):
    model, device = load_rmbg_model()

    transform_image = transforms.Compose([
        transforms.Resize(RMBG_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rgb_image = image.convert("RGB")
    input_tensor = transform_image(rgb_image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    mask = transforms.ToPILImage()(preds[0].squeeze()).resize(rgb_image.size)

    result = rgb_image.copy()
    result.putalpha(mask)
    return result


def image_to_avatar(image: Image):
    print(f"Processing image of size {image.size}")
    pose = extract_pose(image)

    cropped_image = crop_person(image, pose)
    print(f"Cropped image of size {cropped_image.size}")
    if cropped_image.size[0] < CROP_RESOLUTION or cropped_image.size[1] < CROP_RESOLUTION:
        raise ValueError(f"Image is too small. Cropped region should be at least {CROP_RESOLUTION}x{CROP_RESOLUTION}")

    sfw = is_safe_for_work(image)
    print("Is safe for work", sfw)
    if not sfw:
        raise ValueError("Image is not safe for work")

    masked_image = remove_image_background(cropped_image)
    # paste on green background
    green_screen = Image.new("RGB", masked_image.size, "green")
    masked_image = Image.composite(masked_image, green_screen, masked_image)

    return cropped_image, masked_image, extract_pose(image)


if __name__ == "__main__":
    assets_path = Path(__file__).parent.parent / "assets"
    examples = ["flux", "stock", "amit"]
    for example in examples:
        example_dir = assets_path / "examples" / example
        img = Image.open(example_dir / "source.jpg")
        ci, mi, example_pose = image_to_avatar(img)
        ci.save(example_dir / "avatar.jpg")
        mi.save(example_dir / "masked.png")
        with open(example_dir / "pose.pose", "wb") as f:
            example_pose.write(f)
