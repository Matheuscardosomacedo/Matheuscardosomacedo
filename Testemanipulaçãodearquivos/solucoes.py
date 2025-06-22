"""
SOLUÇÕES DOS EXERCÍCIOS DE MANIPULAÇÃO DE ARQUIVOS
==================================================

Aqui estão as soluções para os exercícios. Tente resolver primeiro!
"""

import os
import datetime
from collections import Counter

def solucao_exercicio_1():
    """Solução do Exercício 1: Criar arquivo de notas"""
    print("=== SOLUÇÃO EXERCÍCIO 1 ===")
    
    arquivo = open('notas.txt', 'w', encoding='utf-8')
    arquivo.write("Matemática: 8.5\n")
    arquivo.write("Português: 7.0\n")
    arquivo.write("História: 9.0\n")
    arquivo.write("Ciências: 8.0\n")
    arquivo.close()
    
    print("Arquivo 'notas.txt' criado com sucesso!")


def solucao_exercicio_2():
    """Solução do Exercício 2: Ler e calcular média"""
    print("\n=== SOLUÇÃO EXERCÍCIO 2 ===")
    
    try:
        arquivo = open('notas.txt', 'r', encoding='utf-8')
        notas = []
        
        for linha in arquivo:
            # Extrai o número da linha (ex: "Matemática: 8.5" -> 8.5)
            partes = linha.strip().split(': ')
            if len(partes) == 2:
                nota = float(partes[1])
                notas.append(nota)
        
        arquivo.close()
        
        if notas:
            media = sum(notas) / len(notas)
            print(f"Notas: {notas}")
            print(f"Média: {media:.2f}")
        else:
            print("Nenhuma nota encontrada!")
            
    except FileNotFoundError:
        print("Arquivo 'notas.txt' não encontrado!")


def solucao_exercicio_3():
    """Solução do Exercício 3: Criar agenda de contatos"""
    print("\n=== SOLUÇÃO EXERCÍCIO 3 ===")
    
    contatos = [
        "João Silva - (11) 99999-9999 - joao@email.com",
        "Maria Santos - (21) 88888-8888 - maria@email.com",
        "Pedro Costa - (31) 77777-7777 - pedro@email.com",
        "Ana Oliveira - (41) 66666-6666 - ana@email.com",
        "Carlos Lima - (51) 55555-5555 - carlos@email.com"
    ]
    
    arquivo = open('contatos.txt', 'w', encoding='utf-8')
    for contato in contatos:
        arquivo.write(contato + "\n")
    arquivo.close()
    
    print("Arquivo 'contatos.txt' criado com sucesso!")


def solucao_exercicio_4():
    """Solução do Exercício 4: Buscar contato"""
    print("\n=== SOLUÇÃO EXERCÍCIO 4 ===")
    
    nome_busca = input("Digite o nome para buscar: ").lower()
    
    try:
        arquivo = open('contatos.txt', 'r', encoding='utf-8')
        encontrado = False
        
        for linha in arquivo:
            if nome_busca in linha.lower():
                print(f"Contato encontrado: {linha.strip()}")
                encontrado = True
        
        if not encontrado:
            print("Contato não encontrado!")
        
        arquivo.close()
        
    except FileNotFoundError:
        print("Arquivo 'contatos.txt' não encontrado!")


def solucao_exercicio_5():
    """Solução do Exercício 5: Log de atividades"""
    print("\n=== SOLUÇÃO EXERCÍCIO 5 ===")
    
    def registrar_log(funcao_nome):
        agora = datetime.datetime.now()
        data_hora = agora.strftime("%Y-%m-%d %H:%M:%S")
        
        with open('log.txt', 'a', encoding='utf-8') as arquivo:
            arquivo.write(f"{data_hora} - Função executada: {funcao_nome}\n")
    
    # Exemplo de uso
    registrar_log("teste_funcao")
    print("Log registrado em 'log.txt'")


def solucao_exercicio_6():
    """Solução do Exercício 6: Backup de arquivo"""
    print("\n=== SOLUÇÃO EXERCÍCIO 6 ===")
    
    def fazer_backup(nome_arquivo):
        if not os.path.exists(nome_arquivo):
            print(f"Arquivo '{nome_arquivo}' não encontrado!")
            return
        
        # Cria nome do backup com data
        data_atual = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_backup = f"{nome_arquivo}_backup_{data_atual}.txt"
        
        # Copia o conteúdo
        with open(nome_arquivo, 'r', encoding='utf-8') as original:
            with open(nome_backup, 'w', encoding='utf-8') as backup:
                backup.write(original.read())
        
        print(f"Backup criado: {nome_backup}")
    
    # Exemplo de uso
    fazer_backup('notas.txt')


def solucao_exercicio_7():
    """Solução do Exercício 7: Estatísticas de arquivo"""
    print("\n=== SOLUÇÃO EXERCÍCIO 7 ===")
    
    def analisar_arquivo(nome_arquivo):
        try:
            with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
                conteudo = arquivo.read()
                linhas = arquivo.readlines()
                
                # Conta linhas
                num_linhas = len(linhas)
                
                # Conta palavras
                palavras = conteudo.split()
                num_palavras = len(palavras)
                
                # Conta caracteres
                num_caracteres = len(conteudo)
                
                # Palavra mais frequente
                if palavras:
                    contador = Counter(palavras)
                    palavra_mais_frequente = contador.most_common(1)[0]
                
                print(f"Estatísticas do arquivo '{nome_arquivo}':")
                print(f"- Linhas: {num_linhas}")
                print(f"- Palavras: {num_palavras}")
                print(f"- Caracteres: {num_caracteres}")
                if palavras:
                    print(f"- Palavra mais frequente: '{palavra_mais_frequente[0]}' ({palavra_mais_frequente[1]} vezes)")
                    
        except FileNotFoundError:
            print(f"Arquivo '{nome_arquivo}' não encontrado!")
    
    # Exemplo de uso
    analisar_arquivo('notas.txt')


def solucao_exercicio_8():
    """Solução do Exercício 8: Sistema de configuração"""
    print("\n=== SOLUÇÃO EXERCÍCIO 8 ===")
    
    # Criar arquivo de configuração
    config = {
        'nome_usuario': 'João Silva',
        'idioma': 'pt-BR',
        'tema': 'claro',
        'tamanho_fonte': '14'
    }
    
    # Escrever configuração
    with open('config.txt', 'w', encoding='utf-8') as arquivo:
        for chave, valor in config.items():
            arquivo.write(f"{chave}={valor}\n")
    
    print("Arquivo 'config.txt' criado!")
    
    # Ler configuração
    config_lida = {}
    with open('config.txt', 'r', encoding='utf-8') as arquivo:
        for linha in arquivo:
            chave, valor = linha.strip().split('=')
            config_lida[chave] = valor
    
    print("\nConfigurações carregadas:")
    for chave, valor in config_lida.items():
        print(f"- {chave}: {valor}")


def main():
    """Função principal que executa todas as soluções"""
    print("=== SOLUÇÕES DOS EXERCÍCIOS ===\n")
    
    print("⚠️  IMPORTANTE: Tente resolver os exercícios primeiro!")
    print("Só consulte as soluções depois de tentar.\n")
    
    # Descomente as linhas abaixo para ver as soluções:
    
    # solucao_exercicio_1()
    # solucao_exercicio_2()
    # solucao_exercicio_3()
    # solucao_exercicio_4()
    # solucao_exercicio_5()
    # solucao_exercicio_6()
    # solucao_exercicio_7()
    # solucao_exercicio_8()
    
    print("\n=== CONCEITOS IMPORTANTES APRENDIDOS ===")
    print("✅ Abertura e fechamento de arquivos")
    print("✅ Diferentes modos de acesso (r, w, a)")
    print("✅ Tratamento de exceções")
    print("✅ Uso do 'with' statement")
    print("✅ Manipulação de strings e dados")
    print("✅ Trabalho com datas e horários")
    print("✅ Verificação de existência de arquivos")


if __name__ == "__main__":
    main() 