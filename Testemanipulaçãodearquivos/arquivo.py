import os

def criar_arquivo():
    """Cria um arquivo de texto e escreve conteúdo nele"""
    print("=== CRIANDO ARQUIVO ===")
    
    # Modo 'w' (write) - cria um novo arquivo ou sobrescreve se já existir
    arquivo = open('teste.txt', 'w', encoding='utf-8')
    
    # Escrevendo conteúdo no arquivo
    arquivo.write("Olá! Este é meu primeiro arquivo em Python!\n")
    arquivo.write("Segunda linha do arquivo.\n")
    arquivo.write("Terceira linha com números: 123\n")
    
    # Sempre fechar o arquivo após usar
    arquivo.close()
    print("Arquivo 'teste.txt' criado com sucesso!")


def ler_arquivo():
    """Lê o conteúdo de um arquivo"""
    print("\n=== LENDO ARQUIVO ===")
    
    try:
        # Modo 'r' (read) - lê o arquivo
        arquivo = open('teste.txt', 'r', encoding='utf-8')
        
        # Lendo todo o conteúdo de uma vez
        conteudo = arquivo.read()
        print("Conteúdo completo do arquivo:")
        print(conteudo)
        
        arquivo.close()
        
    except FileNotFoundError:
        print("Erro: Arquivo não encontrado!")


def ler_arquivo_linha_por_linha():
    """Lê o arquivo linha por linha"""
    print("\n=== LENDO LINHA POR LINHA ===")
    
    try:
        arquivo = open('teste.txt', 'r', encoding='utf-8')
        
        # Lendo linha por linha
        for numero_linha, linha in enumerate(arquivo, 1):
            print(f"Linha {numero_linha}: {linha.strip()}")
        
        arquivo.close()
        
    except FileNotFoundError:
        print("Erro: Arquivo não encontrado!")


def adicionar_conteudo():
    """Adiciona conteúdo ao final do arquivo (append)"""
    print("\n=== ADICIONANDO CONTEÚDO ===")
    
    # Modo 'a' (append) - adiciona ao final do arquivo
    arquivo = open('teste.txt', 'a', encoding='utf-8')
    
    arquivo.write("\n=== NOVO CONTEÚDO ===\n")
    arquivo.write("Esta linha foi adicionada depois!\n")
    arquivo.write("Outra linha adicionada.\n")
    
    arquivo.close()
    print("Conteúdo adicionado ao arquivo!")


def verificar_se_arquivo_existe():
    """Verifica se um arquivo existe"""
    print("\n=== VERIFICANDO EXISTÊNCIA ===")
    
    if os.path.exists('teste.txt'):
        print("O arquivo 'teste.txt' existe!")
        
        # Obtendo informações do arquivo
        tamanho = os.path.getsize('teste.txt')
        print(f"Tamanho do arquivo: {tamanho} bytes")
        
    else:
        print("O arquivo 'teste.txt' não existe!")


def criar_arquivo_com_dados_estruturados():
    """Cria um arquivo com dados estruturados (como CSV)"""
    print("\n=== CRIANDO ARQUIVO COM DADOS ESTRUTURADOS ===")
    
    # Dados de exemplo
    pessoas = [
        ["João", "25", "São Paulo"],
        ["Maria", "30", "Rio de Janeiro"],
        ["Pedro", "22", "Belo Horizonte"],
        ["Ana", "28", "Salvador"]
    ]
    
    arquivo = open('pessoas.csv', 'w', encoding='utf-8')
    
    # Escrevendo cabeçalho
    arquivo.write("Nome,Idade,Cidade\n")
    
    # Escrevendo dados
    for pessoa in pessoas:
        linha = ",".join(pessoa)
        arquivo.write(linha + "\n")
    
    arquivo.close()
    print("Arquivo 'pessoas.csv' criado com sucesso!")


def ler_arquivo_csv():
    """Lê um arquivo CSV"""
    print("\n=== LENDO ARQUIVO CSV ===")
    
    try:
        arquivo = open('pessoas.csv', 'r', encoding='utf-8')
        
        for linha in arquivo:
            # Removendo quebra de linha e dividindo por vírgula
            dados = linha.strip().split(',')
            print(f"Nome: {dados[0]}, Idade: {dados[1]}, Cidade: {dados[2]}")
        
        arquivo.close()
        
    except FileNotFoundError:
        print("Erro: Arquivo não encontrado!")


def usar_with_statement():
    """Demonstra o uso do 'with' statement (recomendado)"""
    print("\n=== USANDO 'WITH' STATEMENT ===")
    
    # O 'with' fecha automaticamente o arquivo
    with open('teste.txt', 'r', encoding='utf-8') as arquivo:
        conteudo = arquivo.read()
        print("Conteúdo lido com 'with':")
        print(conteudo)
    
    # Arquivo já está fechado automaticamente
    print("Arquivo fechado automaticamente!")


def main():
    """Função principal que demonstra todas as operações"""
    print("=== APRENDENDO MANIPULAÇÃO DE ARQUIVOS EM PYTHON ===\n")
    
    # 1. Criar arquivo
    criar_arquivo()
    
    # 2. Verificar se existe
    verificar_se_arquivo_existe()
    
    # 3. Ler arquivo completo
    ler_arquivo()
    
    # 4. Ler linha por linha
    ler_arquivo_linha_por_linha()
    
    # 5. Adicionar conteúdo
    adicionar_conteudo()
    
    # 6. Ler novamente para ver as mudanças
    print("\n=== ARQUIVO APÓS ADICIONAR CONTEÚDO ===")
    ler_arquivo()
    
    # 7. Usar 'with' statement
    usar_with_statement()
    
    # 8. Trabalhar com dados estruturados
    criar_arquivo_com_dados_estruturados()
    ler_arquivo_csv()
    
    print("\n=== FIM DO EXEMPLO ===")
    print("Agora você sabe manipular arquivos em Python!")


if __name__ == "__main__":
    main()
    