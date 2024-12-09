import os
from torch.nn.functional import cosine_similarity
from cam_preprocess import capture_face, preprocess_image
import torch

def verify_face(model, username, image_size=128, base_path="users/", embedding_path="embedding.pt", threshold=0.8):
    """
    Verifies the captured face for the given username by comparing the new embedding with the stored one.
    """
    user_path = os.path.join(base_path, username)
    user_embedding_path = os.path.join(user_path, embedding_path)
    
    if not os.path.exists(user_embedding_path):
        print(f"No registered face found for {username}. Please register first.")
        return

    capture_face()  # Capture a new face image
    image = preprocess_image("captured_face.jpg", image_size=image_size)  # Preprocess the image
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        new_embedding = model(image)  # Get the new face embedding
    
    registered_embedding = torch.load(user_embedding_path)  # Load the registered embedding
    similarity = cosine_similarity(new_embedding, registered_embedding).item()  # Compare embeddings
    
    print(f"Similarity: {similarity}")
    if similarity > threshold:  # If the similarity is greater than the threshold, verification is successful
        print(f"Face Verified for {username}!")
    else:
        print("Face Not Verified!")
