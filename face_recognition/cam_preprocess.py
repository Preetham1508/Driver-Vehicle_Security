import cv2
from PIL import Image
import numpy as np
import torch

def capture_face(image_path="captured_face.jpg"):
    cap = cv2.VideoCapture(0)
    print("Press 's' to capture an image, or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")
            break
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def preprocess_image(image_path, image_size=128):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((image_size, image_size))
    image = np.array(image) / 127.5 - 1.0
    image = np.transpose(image, (2, 0, 1))  # Convert to CHW format for PyTorch
    return torch.tensor(image).unsqueeze(0).float()  # Add batch dimension
