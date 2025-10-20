from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import os

# --- seletor de arquivos (Tkinter) ---
import tkinter as tk
from tkinter import filedialog

def escolher_imagens():
    root = tk.Tk()
    root.withdraw()  # não mostrar janela principal
    paths = filedialog.askopenfilenames(
        title="Escolha 1 ou mais imagens",
        filetypes=[
            ("Imagens", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
            ("Todos os arquivos", "*.*"),
        ],
    )
    root.update()
    root.destroy()
    return list(paths)

def imread_unicode_safe(path):
    """
    Lê imagem lidando bem com caminhos com acentos/UTF-8 (Windows-friendly).
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        # fallback
        return cv2.imread(path)

# ==========================
#   Configurações iniciais
# ==========================

# Mantendo variáveis da sua “estrutura”
track_history = defaultdict(lambda: [])
seguir = False          # Para imagem estática, tracking não faz sentido; deixei False
deixar_rastro = False   # idem
janela_nome = "Detecção YOLOv8"

# Modelo treinado personalizado para detectar garrafa_plastico
model = YOLO("runs/detect/train10/weights/best.pt")

# ==========================
#   Fluxo com imagens
# ==========================

imagens = escolher_imagens()

if not imagens:
    print("Nenhuma imagem selecionada. Encerrando.")
    exit(0)

for path in imagens:
    img = imread_unicode_safe(path)
    if img is None:
        print(f"[AVISO] Não consegui carregar: {path}")
        continue

    # Inferência
    # Com imagem estática, usar .predict é suficiente
    results = model.predict(source=img, conf=0.25, imgsz=640, verbose=False)

    # results é uma lista (1 por imagem). Pegamos o primeiro:
    res = results[0]

    # Desenhar anotações na imagem
    anotada = res.plot()

    # (Opcional) coletar contagem de detecções por classe para título
    class_counts = {}
    if res.boxes is not None and res.boxes.cls is not None:
        classes = res.boxes.cls.cpu().numpy().astype(int).tolist()
        names = model.names
        for c in classes:
            nome = names.get(c, str(c))
            class_counts[nome] = class_counts.get(nome, 0) + 1

    # Preparar título da janela
    base = os.path.basename(path)
    if class_counts:
        resumo = ", ".join([f"{k}:{v}" for k, v in class_counts.items()])
        titulo = f"{base}  |  {resumo}"
    else:
        titulo = f"{base}  |  sem detecções"

    # Mostrar
    cv2.imshow(janela_nome, anotada)
    cv2.setWindowTitle(janela_nome, titulo)

    # Controles:
    # - 'q' : sair de tudo
    # - 's' : salvar a imagem anotada ao lado do arquivo original (sufixo _yolo)
    # - qualquer outra tecla: vai para a próxima imagem
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):
        raiz, ext = os.path.splitext(path)
        out_path = f"{raiz}_yolo{ext}"
        # garantir codificação correta
        try:
            ok, buf = cv2.imencode(ext if ext else ".jpg", anotada)
            if ok:
                buf.tofile(out_path)  # preserva acentos no Windows
                print(f"Salvo: {out_path}")
            else:
                print("Falha ao codificar a imagem; tentando fallback...")
                cv2.imwrite(out_path, anotada)
                print(f"Salvo (fallback): {out_path}")
        except Exception as e:
            print(f"Erro ao salvar {out_path}: {e}")
    # caso contrário, continua para a próxima imagem

cv2.destroyAllWindows()
print("finalizado.")
