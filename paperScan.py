import cv2
import numpy as np
import io
from flask import Flask, request, jsonify, send_file
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def preprocess_image(image):
    """ Convert image to grayscale and apply Gaussian blur. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection
    return edges

def find_paper_contour(edges):
    """ Find the largest contour (assumed to be the paper). """
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:  # Paper should be a quadrilateral
            return approx

    return None

def warp_paper(image, paper_contour):
    """ Apply perspective transform to get a top-down view of the document. """
    rect = np.array([point[0] for point in paper_contour], dtype="float32")

    # Order points: top-left, top-right, bottom-right, bottom-left
    s = rect.sum(axis=1)
    diff = np.diff(rect, axis=1)
    
    ordered_rect = np.array([
        rect[np.argmin(s)],  # Top-left
        rect[np.argmin(diff)],  # Top-right
        rect[np.argmax(s)],  # Bottom-right
        rect[np.argmax(diff)]   # Bottom-left
    ], dtype="float32")

    widthA = np.linalg.norm(ordered_rect[2] - ordered_rect[3])
    widthB = np.linalg.norm(ordered_rect[1] - ordered_rect[0])
    heightA = np.linalg.norm(ordered_rect[1] - ordered_rect[2])
    heightB = np.linalg.norm(ordered_rect[0] - ordered_rect[3])

    max_width = max(int(widthA), int(widthB))
    max_height = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    image = np.array(image)

    edges = preprocess_image(image)
    paper_contour = find_paper_contour(edges)

    if paper_contour is None:
        return jsonify({"error": "No paper detected"}), 400

    cropped_image = warp_paper(image, paper_contour)

    # Convert to PIL format and send as response
    _, buffer = cv2.imencode(".jpg", cropped_image)
    return send_file(io.BytesIO(buffer), mimetype="image/jpeg")




if __name__ == "__main__":
    app.run(debug=True)
