from ultralytics import YOLO
import cv2
import os

def testar_modelo_personalizado():
    """
    Testa o modelo treinado personalizado para verificar se detecta apenas garrafa_plastico
    """
    
    print("🔍 Testando modelo personalizado...")
    
    # Carrega o modelo treinado
    try:
        model = YOLO("runs/detect/train/weights/best.pt")
        print("✅ Modelo personalizado carregado com sucesso!")
        
        # Mostra as classes do modelo
        print(f"📋 Classes do modelo: {model.names}")
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return
    
    # Procura por uma imagem de teste no dataset
    dataset_path = "datasets/lixo_praia/images/train"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Diretório não encontrado: {dataset_path}")
        return
    
    # Pega a primeira imagem disponível
    imagens = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not imagens:
        print("❌ Nenhuma imagem encontrada no dataset")
        return
    
    # Testa com a primeira imagem
    imagem_teste = os.path.join(dataset_path, imagens[0])
    print(f"🖼️ Testando com: {imagens[0]}")
    
    try:
        # Faz a predição
        results = model.predict(source=imagem_teste, conf=0.25, verbose=True)
        
        # Analisa os resultados
        res = results[0]
        
        if res.boxes is not None and len(res.boxes) > 0:
            classes_detectadas = res.boxes.cls.cpu().numpy().astype(int).tolist()
            nomes_detectados = [model.names[c] for c in classes_detectadas]
            confiancas = res.boxes.conf.cpu().numpy().tolist()
            
            print(f"\n🎯 Detecções encontradas:")
            for nome, conf in zip(nomes_detectados, confiancas):
                print(f"  - {nome}: {conf:.2f}")
            
            # Verifica se detectou apenas garrafa_plastico
            classes_unicas = set(nomes_detectados)
            if classes_unicas == {'garrafa_plastico'}:
                print("\n✅ SUCESSO! Modelo detectando apenas 'garrafa_plastico' como esperado!")
            else:
                print(f"\n⚠️ ATENÇÃO! Modelo detectou outras classes: {classes_unicas}")
                print("Isso pode indicar que o modelo não foi treinado corretamente ou está usando pesos errados.")
        else:
            print("\n🔍 Nenhuma detecção encontrada na imagem de teste")
            
        # Salva imagem com detecções
        img_anotada = res.plot()
        output_path = "teste_deteccao_resultado.jpg"
        cv2.imwrite(output_path, img_anotada)
        print(f"\n💾 Resultado salvo em: {output_path}")
        
    except Exception as e:
        print(f"❌ Erro durante a predição: {e}")

def comparar_modelos():
    """
    Compara o modelo base com o modelo treinado
    """
    print("\n🔄 Comparando modelos...")
    
    try:
        # Modelo base (COCO)
        modelo_base = YOLO("yolov8n.pt")
        print(f"📋 Classes modelo base (primeiras 10): {list(modelo_base.names.values())[:10]}")
        
        # Modelo treinado
        modelo_treinado = YOLO("runs/detect/train4/weights/best.pt")
        print(f"📋 Classes modelo treinado: {modelo_treinado.names}")
        
        print("\n💡 O modelo base tem 80 classes do COCO dataset")
        print("💡 O modelo treinado deve ter apenas 1 classe: garrafa_plastico")
        
    except Exception as e:
        print(f"❌ Erro ao comparar modelos: {e}")

def main():
    print("🚀 Iniciando teste do modelo YOLO personalizado...\n")
    
    # Testa o modelo personalizado
    testar_modelo_personalizado()
    
    # Compara os modelos
    comparar_modelos()
    
    print("\n✨ Teste concluído!")

if __name__ == "__main__":
    main()