import cv2
import imutils
import numpy as np
import io
from flask import Flask, request, jsonify, send_file
from PIL import Image
from flask_cors import CORS
from skimage.filters import threshold_local
from utils import four_point_perspective_transform
from transform import perspective_transform

app = Flask(__name__)
CORS(app)

def process_image(image):
    copy = image.copy()
    ratio = image.shape[0] / 500.0
    img_resize = imutils.resize(image, height=500)

    gray_image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edged_img = cv2.Canny(blurred_image, 75, 200)

    cnts, _ = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    doc = None  # Ensure doc is defined

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc = approx
            break

    if doc is None:
        return None  # Return None if no document contour was found

    p = []
    for d in doc:
        tuple_point = tuple(d[0])
        cv2.circle(img_resize, tuple_point, 3, (0, 0, 255), 4)
        p.append(tuple_point)

    # Apply Perspective Transform
    warped_image = perspective_transform(copy, doc.reshape(4, 2) * ratio)
    
    # ---------------- Sharpening ----------------
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    sharpened_image = cv2.filter2D(warped_image, -1, sharpening_kernel)  # Apply sharpening filter

    return sharpened_image  # Return the processed image


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    
    try:
        image = Image.open(file).convert("RGB")  # Convert to RGB to avoid mode issues
        image = np.array(image)
    except Exception as e:
        return jsonify({"error": "Invalid image file"}), 400

    scanned_image = process_image(image)

    if scanned_image is None:
        return jsonify({"error": "Could not detect a document"}), 400

    pil_image = Image.fromarray(scanned_image)
    img_io = io.BytesIO()
    pil_image.save(img_io, format="JPEG")
    img_io.seek(0)

    return send_file(img_io, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)



# def process_image(image):
#     """Process the image to scan the document."""
#     image_copy = image.copy()
#     image = cv2.resize(image, (1500, 800))
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
#     image_edge = cv2.Canny(image_gray, 75, 200)

#     # Find contours
#     cnts, _ = cv2.findContours(image_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

#     screenCnt = None
#     for c in cnts:
#         peri = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#         if len(approx) == 4:
#             screenCnt = approx
#             break

#     if screenCnt is None:
#         return None

#     # Get the perspective transformation
#     warped_image = four_point_perspective_transform(image, screenCnt.reshape(4, 2))

#     # Convert to grayscale and apply threshold for scanned effect
#     warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
#     T = threshold_local(warped_image, 11, offset=10, method="gaussian")
#     warped_image = (warped_image > T).astype("uint8") * 255

#     return warped_image

# @app.route("/upload", methods=["POST"])
# def upload():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
    
#     # Read the image using PIL
#     image = Image.open(file)
#     image = np.array(image)

#     # Process the image
#     scanned_image = process_image(image)

#     # -------------------------------------
#     if scanned_image is None:
#         return jsonify({"error": "Could not detect a document"}), 400

#     # Convert back to PIL image
#     pil_image = Image.fromarray(scanned_image)

#     # Save to buffer
#     img_io = io.BytesIO()
#     pil_image.save(img_io, format="PNG")
#     img_io.seek(0)

#     return send_file(img_io, mimetype="image/png")
# # -----------------------------------

# if __name__ == "__main__":
#     app.run(debug=True)




# -------------------------------------------------
# import cv2
# import numpy as np
# import io
# from flask import Flask, request, jsonify, send_file
# from PIL import Image
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# def preprocess_image(image):
#     """ Enhanced preprocessing for document detection """
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply adaptive thresholding
#     thresh = cv2.adaptiveThreshold(
#         gray, 255, 
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#         cv2.THRESH_BINARY, 11, 2
#     )
    
#     # Apply Gaussian blur
#     blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    
#     # Edge detection
#     edges = cv2.Canny(blurred, 30, 100)
    
#     return image, edges

# def find_document_contour(edges):
#     """ Detect document contour """
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     valid_contours = []
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         perimeter = cv2.arcLength(contour, True)
        
#         if area > 500:  # Minimum area to filter noise
#             approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
#             if len(approx) == 4:
#                 # Calculate aspect ratio
#                 x, y, w, h = cv2.boundingRect(approx)
#                 aspect_ratio = float(w) / h
                
#                 if 0.5 <= aspect_ratio <= 2.0:
#                     valid_contours.append((area, approx))
    
#     # Return largest contour
#     return max(valid_contours, key=lambda x: x[0])[1] if valid_contours else None

# def order_points(pts):
#     """ Order points for perspective transform """
#     rect = np.zeros((4, 2), dtype="float32")
    
#     # Top-left will have smallest sum, bottom-right largest sum
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
    
#     # Top-right will have smallest difference, bottom-left largest difference
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
    
#     return rect

# def four_point_transform(image, pts):
#     """ Apply perspective transform """
#     rect = order_points(pts)
#     (tl, tr, br, bl) = rect

#     # Compute width of new image
#     widthA = np.linalg.norm(br - bl)
#     widthB = np.linalg.norm(tr - tl)
#     max_width = max(int(widthA), int(widthB))

#     # Compute height of new image
#     heightA = np.linalg.norm(tr - br)
#     heightB = np.linalg.norm(tl - bl)
#     max_height = max(int(heightA), int(heightB))

#     # Destination points
#     dst = np.array([
#         [0, 0],
#         [max_width - 1, 0],
#         [max_width - 1, max_height - 1],
#         [0, max_height - 1]
#     ], dtype="float32")

#     # Compute perspective transform matrix and apply
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(image, M, (max_width, max_height))

#     return warped

# @app.route("/upload", methods=["POST"])
# def upload():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     image = Image.open(file).convert("RGB")
#     img = np.array(image)

#     # Resize image
#     img = cv2.resize(img, (int(480*2), int(640*2)))

#     # Preprocess
#     preprocessed, edges = preprocess_image(img)

#     # Find document contour
#     document_contour = find_document_contour(edges)

#     if document_contour is None:
#         return jsonify({"error": "No document detected"}), 400

#     # Transform image
#     warped = four_point_transform(preprocessed, document_contour.reshape(4, 2))

#     # Convert to PIL format and send as response
#     _, buffer = cv2.imencode(".jpg", warped)
#     return send_file(io.BytesIO(buffer), mimetype="image/jpeg")

# if __name__ == "__main__":
#     app.run(debug=True)