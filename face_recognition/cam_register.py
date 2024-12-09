import os
from cam_preprocess import capture_face, preprocess_image
import torch

def register_face(model, username, image_size=128, base_path="users/", embedding_path="embedding.pt"):
    """
    Registers a new face for the given username. The face embedding is saved in a folder named after the username.
    """
    user_path = os.path.join(base_path, username)
    if not os.path.exists(user_path):
        os.makedirs(user_path)  # Create a directory for the user if it doesn't exist

    capture_face()  # Capture face image
    image = preprocess_image("captured_face.jpg", image_size=image_size)  # Preprocess the image
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        embedding = model(image)  # Get the face embedding from the model
    
    user_embedding_path = os.path.join(user_path, embedding_path)
    torch.save(embedding, user_embedding_path)  # Save the embedding to the user's folder
    print(f"Face embedding for {username} saved as {user_embedding_path}")
