from torch.nn.functional import cosine_similarity
from cam_preprocess import capture_face
from cam_preprocess import preprocess_image
import torch

def verify_face(model, image_size=128, embedding_path="registered_face.pt", threshold=0.8):
    capture_face()
    image = preprocess_image("captured_face.jpg", image_size=image_size)
    model.eval()
    with torch.no_grad():
        new_embedding = model(image)
    registered_embedding = torch.load(embedding_path)
    similarity = cosine_similarity(new_embedding, registered_embedding).item()
    print(f"Similarity: {similarity}")
    if similarity > threshold:
        print("Face Verified!")
    else:
        print("Face Not Verified!")
