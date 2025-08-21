import os
import traceback
from pathlib import Path

from flask import Flask, Response, abort, jsonify, make_response, request
from PIL import Image

from human_avatar.image_to_avatar import image_to_avatar as image_to_human_avatar

app = Flask(__name__)


def resolve_path(uri: str):
    # Map gs:// URIs to the gcsfuse mount point, or return as-is
    return uri.replace("gs://", "/mnt/")


@app.errorhandler(Exception)
def handle_exception(e):
    print("Exception", e)
    traceback.print_exc()

    code = e.code if hasattr(e, "code") else 500
    message = str(e)
    print("HTTP exception", code, message)

    return make_response(jsonify(message=message, code=code), code)


@app.route("/", methods=['POST'])
def image_to_avatar():
    # Get request parameters
    output = request.form.get("output")
    if not output:
        abort(make_response(jsonify(message="Missing `output` body property"), 400))

    output_directory = Path(resolve_path(output))

    if 'file' not in request.files:
        abort(make_response(jsonify(message="File named 'file' is missing"), 400))

    file = request.files['file']
    image = Image.open(file)
    print("Image size", image.size)

    cropped, masked, pose = image_to_human_avatar(image)

    output_directory.mkdir(parents=True, exist_ok=True)
    image.save(output_directory / "original.png")
    cropped.save(output_directory / "cropped.png")
    masked.save(output_directory / "masked.png")

    with (output_directory / "pose.pose").open("wb") as pose_file:
        pose.write(pose_file)

    return Response(status=201)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
