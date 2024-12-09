from cam_preprocess import capture_face, preprocess_image
import torch

def register_face(model, image_size=128, embedding_path="registered_face.pt"):
    capture_face()
    image = preprocess_image("captured_face.jpg", image_size=image_size)
    model.eval()
    with torch.no_grad():
        embedding = model(image)
    torch.save(embedding, embedding_path)
    print(f"Face embedding saved as {embedding_path}")
