import cv2
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
import pickle

detector = MTCNN()
embedding_model = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_frame)

    if detections:
        x, y, width, height = detections[0]['box']
        face = rgb_frame[y:y + height, x:x + width]
        face = cv2.resize(face, (160, 160))
        face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        embedding = embedding_model(face).detach().numpy()
        return embedding

def save_embedding(embedding, user_id="user1"):
    with open(f"data/embeddings/{user_id}.pkl", "wb") as f:
        pickle.dump(embedding, f)
    print(f"Embedding salvo para o ID: {user_id}")

def capture_and_save_embedding(user_id="user1"):
    cap = cv2.VideoCapture(0)
    print("Pressione 's' para salvar o embedding do rosto.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            embedding = get_embedding(frame)
            if embedding is not None:
                save_embedding(embedding, user_id)
                print(f"Embedding salvo para {user_id}")
            else:
                print("Nenhum rosto detectado. Tente novamente.")
            break

    cap.release()
    cv2.destroyAllWindows()
