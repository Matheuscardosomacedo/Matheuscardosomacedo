def main():

    vwkh = float(0.0)
    ncons = int(0.0)
    qtdcons = float (0.0)
    cod = str ()
    primeiro = True

    somaR = 0
    somaC = 0
    somaI = 0
    conte = 0

  #Processamento de dados

    vwkh = float(input('preço do kWh consumido: '))
    ncons = int(input('número do consumidor: '))

    while (ncons > 0):

        
      
        qtdcons = int(input('quantidade de kWh consumidos durante o mês: '))
        cod = str(input('código do tipo de consumidor R/C/I: '))
        conte = conte + 1
        tpg = qtdcons * vwkh

        print ( ncons)
        print ( tpg )

        if ( primeiro ):

            maior = qtdcons
            menor = qtdcons
            primeiro = False
        else:

            if ( maior < qtdcons):
                
                maior = qtdcons
            else:

                
                menor = qtdcons

        if (cod == 'R'):
            somaR = somaR + qtdcons
        else:
            if (cod == 'I'):
                somaI = somaI + qtdcons
            else:
                somaC = somaC + qtdcons
        ncons = int(input('número do consumidor: '))


    print (maior)
    print (menor)
    print (somaR)    
    print (somaC)
    print (somaI)
    print ((somaR+somaI+somaC)/conte)



    return 0


if __name__ == "__main__":
    main()