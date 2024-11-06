from mtcnn import MTCNN
import cv2

# Inicializa o detector MTCNN
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

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
        return

    print("Pressione 'q' para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detecta rostos
        faces = detect_faces(frame)

        # Desenha caixas e atribui IDs
        for i, (x, y, width, height) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {i + 1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Mostra o resultado na tela com tamanho padrão
        cv2.imshow("Facial Detection", frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
