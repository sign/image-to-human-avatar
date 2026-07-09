from functools import cache
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from pose_format.utils.holistic import load_holistic
from torchvision import transforms
from transformers import AutoModelForImageSegmentation, pipeline

CROP_RESOLUTION = 512
RMBG_INPUT_SIZE = (1024, 1024)
DETECTION_RESOLUTION = 512

SHOULDER_LANDMARKS = (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)


def extract_shoulders(image: Image):
    """Returns the (left, right) shoulder positions, normalized to [0, 1]."""
    # BlazePose (body only, lowest complexity) since we only need shoulders for cropping
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=0) as pose_model:
        results = pose_model.process(np.array(image))

    if results.pose_landmarks is None:
        raise ValueError("No pose detected")

    landmarks = results.pose_landmarks.landmark
    return [(landmarks[point].x, landmarks[point].y) for point in SHOULDER_LANDMARKS]


def extract_full_pose(image: Image):
    # Full holistic pose (body, face, hands) for avatar animation
    frames = [np.array(image.convert("RGB"))]
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


def crop_person(image: Image, l_shoulder, r_shoulder):
    center_x = (l_shoulder[0] + r_shoulder[0]) / 2 * image.width
    center_y = (l_shoulder[1] + r_shoulder[1]) / 2 * image.height
    crop_size = 1.25 * abs(l_shoulder[0] - r_shoulder[0]) * image.width

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


def image_to_avatar(image: Image, include_pose=False):
    print(f"Processing image of size {image.size}")
    # BlazePose and the NSFW classifier infer at low resolution internally, so a box-reduced
    # copy feeds both; box averaging keeps landmarks within ~0.2% of full-res detection
    factor = max(1, max(image.size) // DETECTION_RESOLUTION)
    detection_image = image.convert("RGB").reduce(factor)

    l_shoulder, r_shoulder = extract_shoulders(detection_image)

    cropped_image = crop_person(image, l_shoulder, r_shoulder)
    print(f"Cropped image of size {cropped_image.size}")
    if cropped_image.size[0] < CROP_RESOLUTION or cropped_image.size[1] < CROP_RESOLUTION:
        raise ValueError(f"Image is too small. Cropped region should be at least {CROP_RESOLUTION}x{CROP_RESOLUTION}")

    sfw = is_safe_for_work(detection_image)
    print("Is safe for work", sfw)
    if not sfw:
        raise ValueError("Image is not safe for work")

    masked_image = remove_image_background(cropped_image)
    # paste on green background
    green_screen = Image.new("RGB", masked_image.size, "green")
    masked_image = Image.composite(masked_image, green_screen, masked_image)

    full_pose = extract_full_pose(image) if include_pose else None
    return cropped_image, masked_image, full_pose


if __name__ == "__main__":
    assets_path = Path(__file__).parent.parent / "assets"
    examples = ["flux", "stock", "amit"]
    for example in examples:
        example_dir = assets_path / "examples" / example
        img = Image.open(example_dir / "source.jpg")
        ci, mi, example_pose = image_to_avatar(img, include_pose=True)
        ci.save(example_dir / "avatar.jpg")
        mi.save(example_dir / "masked.png")
        with open(example_dir / "pose.pose", "wb") as f:
            example_pose.write(f)
