import os
import sys

def corrigir_labels_latas():
    """
    Corrige as labels das latas, alterando de classe 0 para classe 1
    baseado nos nomes dos arquivos que contêm palavras relacionadas a latas
    """
    
    # Diretório das labels
    script_dir = os.path.dirname(os.path.abspath(__file__))
    diretorio_labels = os.path.join(script_dir, "datasets", "lixo_praia", "labels", "train")
    
    if not os.path.exists(diretorio_labels):
        print(f"❌ Diretório não encontrado: {diretorio_labels}")
        sys.exit(1)
    
    # Palavras-chave que indicam que o arquivo é de uma lata
    palavras_lata = [
        'lata', 'can', 'coca', 'cola', 'pepsi', 'refrigerante', 'soda', 
        'crushed', 'crumpled', 'amassada', 'esmagada', 'drink-can',
        'schweppes', 'tonica'
    ]
    
    arquivos_corrigidos = 0
    arquivos_processados = 0
    
    print("🔧 Iniciando correção das labels das latas...")
    print(f"📁 Diretório: {diretorio_labels}")
    
    # Percorre todos os arquivos .txt no diretório
    for arquivo in os.listdir(diretorio_labels):
        if arquivo.endswith('.txt'):
            caminho_arquivo = os.path.join(diretorio_labels, arquivo)
            nome_arquivo_lower = arquivo.lower()
            
            # Verifica se o nome do arquivo contém palavras relacionadas a latas
            eh_lata = any(palavra in nome_arquivo_lower for palavra in palavras_lata)
            
            if eh_lata:
                arquivos_processados += 1
                
                try:
                    # Lê o conteúdo do arquivo
                    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                        linhas = f.readlines()
                    
                    # Verifica se precisa corrigir (se tem classe 0)
                    linhas_corrigidas = []
                    foi_corrigido = False
                    
                    for linha in linhas:
                        linha = linha.strip()
                        if linha.startswith('0 '):
                            # Altera de classe 0 para classe 1
                            linha_corrigida = '1 ' + linha[2:]
                            linhas_corrigidas.append(linha_corrigida + '\\n')
                            foi_corrigido = True
                        else:
                            linhas_corrigidas.append(linha + '\\n')
                    
                    # Se foi corrigido, salva o arquivo
                    if foi_corrigido:
                        with open(caminho_arquivo, 'w', encoding='utf-8') as f:
                            f.writelines(linhas_corrigidas)
                        
                        print(f"✅ Corrigido: {arquivo}")
                        arquivos_corrigidos += 1
                    else:
                        print(f"ℹ️  Já correto: {arquivo}")
                        
                except Exception as e:
                    print(f"❌ Erro ao processar {arquivo}: {str(e)}")
    
    print(f"\\n📊 Resumo:")
    print(f"🔍 Arquivos de latas encontrados: {arquivos_processados}")
    print(f"✅ Arquivos corrigidos: {arquivos_corrigidos}")
    print(f"ℹ️  Arquivos já corretos: {arquivos_processados - arquivos_corrigidos}")
    
    if arquivos_corrigidos > 0:
        print(f"\\n🎯 Agora você pode treinar o modelo novamente!")
        print(f"   As latas agora estão marcadas como classe 1 (latinha)")
    else:
        print(f"\\n💡 Nenhuma correção foi necessária.")

if __name__ == "__main__":
    corrigir_labels_latas()