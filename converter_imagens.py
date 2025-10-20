import os
import sys
from PIL import Image
import glob

def converter_imagens_problematicas(diretorio_dataset):
    """
    Converte imagens que podem estar causando problemas no treinamento YOLO.
    Especificamente procura por arquivos .jpg que podem estar em formato AVIF
    ou outros formatos não suportados.
    """
    
    # Padrões de busca para imagens
    padroes = [
        os.path.join(diretorio_dataset, "**", "*.jpg"),
        os.path.join(diretorio_dataset, "**", "*.jpeg"),
        os.path.join(diretorio_dataset, "**", "*.png"),
        os.path.join(diretorio_dataset, "**", "*.webp"),
        os.path.join(diretorio_dataset, "**", "*.avif")
    ]
    
    arquivos_convertidos = 0
    arquivos_com_erro = 0
    
    print("🔍 Procurando por imagens problemáticas...")
    
    for padrao in padroes:
        arquivos = glob.glob(padrao, recursive=True)
        
        for arquivo in arquivos:
            try:
                # Tenta abrir a imagem
                with Image.open(arquivo) as img:
                    # Verifica se a imagem está em um formato problemático
                    formato_original = img.format
                    
                    # Se for AVIF ou outro formato problemático, converte para JPG
                    if formato_original in ['AVIF', 'WEBP'] or arquivo.lower().endswith('.avif'):
                        print(f"📸 Convertendo: {os.path.basename(arquivo)} ({formato_original} → JPG)")
                        
                        # Converte para RGB se necessário (para evitar problemas com transparência)
                        if img.mode in ('RGBA', 'LA', 'P'):
                            # Cria um fundo branco
                            fundo_branco = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'P':
                                img = img.convert('RGBA')
                            fundo_branco.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                            img = fundo_branco
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Gera novo nome de arquivo
                        nome_base = os.path.splitext(arquivo)[0]
                        novo_arquivo = nome_base + '.jpg'
                        
                        # Se o arquivo original não era .jpg, remove o original
                        if not arquivo.lower().endswith('.jpg'):
                            # Salva a nova imagem
                            img.save(novo_arquivo, 'JPEG', quality=95)
                            # Remove o arquivo original
                            os.remove(arquivo)
                            print(f"✅ Convertido: {os.path.basename(arquivo)} → {os.path.basename(novo_arquivo)}")
                            
                            # Verifica se há arquivo de label correspondente para arquivos renomeados
                            nome_base_original = os.path.splitext(arquivo)[0]
                            nome_base_novo = os.path.splitext(novo_arquivo)[0]
                            
                            arquivo_label_original = nome_base_original + '.txt'
                            arquivo_label_novo = nome_base_novo + '.txt'
                            
                            if os.path.exists(arquivo_label_original) and arquivo_label_original != arquivo_label_novo:
                                os.rename(arquivo_label_original, arquivo_label_novo)
                                print(f"📝 Label renomeado: {os.path.basename(arquivo_label_original)} → {os.path.basename(arquivo_label_novo)}")
                        else:
                            # Se era .jpg mas estava em formato AVIF, sobrescreve
                            img.save(arquivo, 'JPEG', quality=95)
                            print(f"✅ Corrigido: {os.path.basename(arquivo)}")
                        
                        arquivos_convertidos += 1
                            
            except Exception as e:
                print(f"❌ Erro ao processar {os.path.basename(arquivo)}: {str(e)}")
                arquivos_com_erro += 1
    
    print(f"\n📊 Resumo:")
    print(f"✅ Arquivos convertidos: {arquivos_convertidos}")
    print(f"❌ Arquivos com erro: {arquivos_com_erro}")
    
    if arquivos_convertidos > 0:
        print(f"\n🎉 Conversão concluída! Agora todas as imagens devem estar em formato compatível com YOLO.")
    else:
        print(f"\n💡 Nenhuma imagem precisou ser convertida. Todas já estão em formatos compatíveis.")

def main():
    # Diretório do dataset - usa caminho absoluto baseado na localização do script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    diretorio_dataset = os.path.join(script_dir, "datasets", "lixo_praia")
    
    if not os.path.exists(diretorio_dataset):
        print(f"❌ Diretório não encontrado: {diretorio_dataset}")
        print("Verifique se a pasta datasets/lixo_praia existe no diretório treinando_yolov8")
        sys.exit(1)
    
    print("🚀 Iniciando conversão de imagens para formatos compatíveis com YOLO...")
    print(f"📁 Diretório: {diretorio_dataset}")
    
    converter_imagens_problematicas(diretorio_dataset)

if __name__ == "__main__":
    main()