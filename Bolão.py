import os


def limpatela():
  if os.name == "nt":
    os.system("cls")
  else:
    os.system("clear")


def verificarjogadores(jogadores, c):
  for (nome, cpf) in jogadores:
    if cpf == c:
      return True

  return False


def visualizarjogadores(jogadores):

  if len(jogadores) == 0:
    print("Não há jogadores cadastrados. ")
  else:
    print("jogadores já cadastrados: ")
    for (nome, cpf) in jogadores:
      print(f' jogador: {nome}, CPF do jogador: {cpf}')

  print()


def inserirjogadores(jogadores):
  nome = input("informe o nome do jogador: ")
  cpf = input("informe o cpf do jogador: ")

  if verificarjogadores(jogadores, cpf):
    print("Erro: jogador já cadastrado. ")
  else:
    print("jogador cadastrado com sucesso.")
    jogadores.append((nome, cpf))


def cadastraaposta(jogadores, apostas):
  cpf = lercpfs(jogadores)
  numeros = lerapostas()
  apostas.append((cpf, numeros))


def lercpfs(jogadores):

  n = int(input(' qual a quantidade de jogadores? '))
  cpfs = []
  if n > len(jogadores):
    print(
        "Erro: numero de jogadores ultrapassar o limite de jogadores cadastrados. "
    )

  else:
    while len(cpfs) < n:
      visualizarjogadores(jogadores)
      cpf = input('insira o cpf do jogador: ')
      while (cpf in cpfs) or (not verificarjogadores(jogadores, cpf)):

        print(' ERRO: CPF não cadastrados.')
        cpf = input("Digite outro cpf: ")
      cpfs.append(cpf)
    return cpfs


def lerapostas():
  n = int(input("Quantos numeros no cartão?: "))
  while n < 6 or n > 15:
    print("Erro: digite um numero entre 6 e 15")
    n = int(input("Quantos numeros no cartão?: "))

  numeros = []
  while len(numeros) < n:
    a = int(input("Digite um numero: "))

    while a < 1 or a > 60 or a in numeros:
      print("ERRO: Número invalido")
      a = int(input("Digite um numero: "))
    numeros.append(a)

  return numeros


def visualisarapostas(jogadores, apostas):
  ap = 0
  while ap < len(apostas):
    cpfs, numeros = apostas[ap]

    print(f'aposta {ap+1}:')
    for cpf in cpfs:
      print(f'jogador: {nomejogador(jogadores, cpf)}:')
    print('numeros:', end=" ")
    for numero in numeros:
      print(numero, end=" ")
    ap += 1


def nomejogador(jogadores, cpf):
  for nome, c in jogadores:
    if c == cpf:
      return nome


def lernumerossorteados():
  l = []

  while len(l) < 6:
    x = int(input("Qual o numero sorteado?: "))
    if x < 1 or x > 60 or x in l:
      print("ERRO: numero invalido")
    else:
      l.append(x)
  return l


def contemvencedores(l, l1):
  for n in l:
    if n not in l1:
      return False
  return True


def verificarganhadores(apostas, sorteados):
  b = []
  for i in range(len(apostas)):
    _, n = apostas[i]
    if contemvencedores(sorteados, n):
      b.append(i)
  return b


def listadevencedores(jogadores, apostas, ganhadores, premioporcartão):
  limpatela()
  print("vencedores")
  for i in ganhadores:
    cpf, _ = apostas[i]
    print(f'cartão: {i+1}')
    for c in cpf:
      print(
          f'jogador: {nomejogador(jogadores, c)} - R$ {premioporcartão/len(cpf):,.2f}'
      )


def inserirsorteior(jogadores, apostas):
  sorteados = lernumerossorteados()
  ganhadores = verificarganhadores(apostas, sorteados)
  premio = int(input("Digite o premio total do sorteio: "))
  if len(ganhadores) > 0:
    divpremio = premio / len(ganhadores)
  listadevencedores(jogadores, apostas, ganhadores, divpremio)


def main():

  jogadores = []
  apostas = []

  print("Seja bem vindo ao Bolão!")
  menu = '''  
  Escolha uma opção:
  
  
   1 - Cadastrar jogador
   2 - Lista de jogadores
   3 - Cadastrar Apostas
   4 - Visualizar apostas
   5 - Inserir Sorteio
   6 - sair do programa
  '''

  opçao = input(menu)
  while opçao != "6":
    limpatela()

    if opçao == '1':
      visualizarjogadores(jogadores)
      inserirjogadores(jogadores)
    elif opçao == '2':
      visualizarjogadores(jogadores)
    elif opçao == '3':
      visualizarjogadores(jogadores)
      cadastraaposta(jogadores, apostas)
    elif opçao == '4':
      print(visualisarapostas(jogadores, apostas))
    elif opçao == '5':
      print(inserirsorteior(jogadores, apostas))
    else:
      print("opição invalida ")
    opçao = input(menu)
  print("Tchau")
  return 0


if __name__ == "__main__":

  main()
