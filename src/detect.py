from mtcnn import MTCNN
import cv2

# Inicialize o detector MTCNN
detector = MTCNN()

def detect_faces(frame):
    # Converte a imagem para RGB (necessário para o MTCNN)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detecta rostos
    detections = detector.detect_faces(rgb_frame)
    
    # Extrai as coordenadas das caixas delimitadoras dos rostos
    faces = []
    for detection in detections:
        x, y, width, height = detection['box']
        faces.append((x, y, width, height))
    
    return faces
