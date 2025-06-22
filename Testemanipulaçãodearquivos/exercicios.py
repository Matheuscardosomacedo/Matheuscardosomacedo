"""
EXERCÍCIOS DE MANIPULAÇÃO DE ARQUIVOS
=====================================

Complete as funções abaixo para praticar manipulação de arquivos.
"""

def exercicio_1():
    """
    EXERCÍCIO 1: Criar um arquivo de notas
    --------------------------------------
    Crie um arquivo chamado 'notas.txt' com as seguintes notas:
    - Matemática: 8.5
    - Português: 7.0
    - História: 9.0
    - Ciências: 8.0
    """
    print("=== EXERCÍCIO 1 ===")
    
    # TODO: Escreva o código aqui
    # Dica: Use open() com modo 'w' e write()
    
    print("Exercício 1 concluído!")


def exercicio_2():
    """
    EXERCÍCIO 2: Ler e calcular média
    --------------------------------
    Leia o arquivo 'notas.txt' criado no exercício 1 e calcule a média das notas.
    """
    print("\n=== EXERCÍCIO 2 ===")
    
    # TODO: Escreva o código aqui
    # Dica: Use open() com modo 'r', leia linha por linha e extraia os números
    
    print("Exercício 2 concluído!")


def exercicio_3():
    """
    EXERCÍCIO 3: Criar agenda de contatos
    -------------------------------------
    Crie um arquivo 'contatos.txt' com pelo menos 5 contatos no formato:
    Nome - Telefone - Email
    """
    print("\n=== EXERCÍCIO 3 ===")
    
    # TODO: Escreva o código aqui
    # Exemplo de formato:
    # João Silva - (11) 99999-9999 - joao@email.com
    
    print("Exercício 3 concluído!")


def exercicio_4():
    """
    EXERCÍCIO 4: Buscar contato
    ---------------------------
    Leia o arquivo 'contatos.txt' e permita que o usuário busque um contato pelo nome.
    """
    print("\n=== EXERCÍCIO 4 ===")
    
    # TODO: Escreva o código aqui
    # Dica: Use input() para pedir o nome e verifique se está na linha
    
    print("Exercício 4 concluído!")


def exercicio_5():
    """
    EXERCÍCIO 5: Log de atividades
    ------------------------------
    Crie um sistema de log que registra as atividades do usuário.
    Cada vez que uma função é chamada, adicione uma entrada no arquivo 'log.txt'
    com data, hora e nome da função.
    """
    print("\n=== EXERCÍCIO 5 ===")
    
    # TODO: Escreva o código aqui
    # Dica: Use modo 'a' para append e import datetime
    
    print("Exercício 5 concluído!")


def exercicio_6():
    """
    EXERCÍCIO 6: Backup de arquivo
    ------------------------------
    Crie uma função que faz backup de um arquivo.
    O backup deve ter o nome original + '_backup' + data atual.
    """
    print("\n=== EXERCÍCIO 6 ===")
    
    # TODO: Escreva o código aqui
    # Dica: Leia o arquivo original e escreva no novo arquivo
    
    print("Exercício 6 concluído!")


def exercicio_7():
    """
    EXERCÍCIO 7: Estatísticas de arquivo
    ------------------------------------
    Crie uma função que analisa um arquivo de texto e retorna:
    - Número total de linhas
    - Número total de palavras
    - Número total de caracteres
    - Palavra mais frequente
    """
    print("\n=== EXERCÍCIO 7 ===")
    
    # TODO: Escreva o código aqui
    # Dica: Use split() para contar palavras e collections.Counter para frequência
    
    print("Exercício 7 concluído!")


def exercicio_8():
    """
    EXERCÍCIO 8: Sistema de configuração
    ------------------------------------
    Crie um arquivo 'config.txt' com configurações do programa:
    - Nome do usuário
    - Idioma (pt-BR, en-US)
    - Tema (claro, escuro)
    - Tamanho da fonte (12, 14, 16)
    
    Depois leia essas configurações e exiba-as.
    """
    print("\n=== EXERCÍCIO 8 ===")
    
    # TODO: Escreva o código aqui
    # Dica: Use formato chave=valor para facilitar a leitura
    
    print("Exercício 8 concluído!")


def main():
    """Função principal que executa todos os exercícios"""
    print("=== EXERCÍCIOS DE MANIPULAÇÃO DE ARQUIVOS ===\n")
    
    print("Para praticar, complete as funções nos comentários TODO!")
    print("Depois execute cada função para testar.\n")
    
    # Descomente as linhas abaixo conforme você completa os exercícios:
    
    # exercicio_1()
    # exercicio_2()
    # exercicio_3()
    # exercicio_4()
    # exercicio_5()
    # exercicio_6()
    # exercicio_7()
    # exercicio_8()
    
    print("\n=== DICAS IMPORTANTES ===")
    print("1. Sempre feche os arquivos após usar (arquivo.close())")
    print("2. Use 'with' statement quando possível")
    print("3. Trate exceções com try/except")
    print("4. Use encoding='utf-8' para caracteres especiais")
    print("5. Teste seus códigos passo a passo!")


if __name__ == "__main__":
    main() 