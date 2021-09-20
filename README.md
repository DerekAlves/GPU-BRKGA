# GPU-BRKGA

## Introdução

Este repositório contém o código fonte da nossa biblioteca em GPU para o algoritmo genético enviesado de chave aleatória, este projeto foi proposto com apoio do CNPQ, FAPEAL e o Instituto de Computação da Universidade Federal de Alagoas. O GPU-BRKGA é fruto de dois ciclos de participação em um projeto de iniciação científica.

Você pode compilar este repositório utilizando o terminal com o NVCC instalado, basta executar o Makefile, se compilado corretamente, você pode executar utilizando:
`./api-usage 'seed' 'gerações'`, por exemplo:
`./api-usage 0 1000`, o algoritmo irá executar com seed 0 e rodar por 1000 gerações.

Autores:

Derek Alves, Davi Oliveira, Bruno Nogueira e Ermeson Andrade

## Como funciona

O  GPU-BRKGA foi implementado em CUDA/C++ e está estruturado a partir de basicamente três classes, sendo estas: GPU-BRKGA, Decoder e Individual.

1)GPU-BRKGA ́e a classe base da nossa API e contém todos os metodos e variaveis necessários para a execução da parte independente do problema.

2)Decoder ́e a classe a ser implementada pelo usuário e precisa conter pelo menos dois métodos: Decode e Init, que serão discutidos adiante. Outros métodos podem ser especificados pelo usuário se necessário.

3)Individual ́e usada para representar um indivíduo em memória da RAM.
Além dessas classes, utilizamos o arquivo kernels.cuh que contém algumas funções e kernels necessários para a API, como por exemplo o kernel Offspring, que é responsável por fazer o cruzamento em GPU.

## Como utilizar

O usuário precisa especificar a classe Decoder, que é o decodificador escolhido para abordar o problema a ser abordado pelo usuário, já indicamos anteiormente que o usuário precisa implementar dois métodos por padrão, esses métodos são:

Decode: É o método que irá decodificar uma população por completo, este método é chamado pelo GPU-BRKGA e o decodificador dado pelo usuário deve tratar a população guardando os valores das aptidões que podem ser calculadas tanto em CPU como em GPU, para isto o usuário deve fornecer o parâmetro gpu_deco ao GPU-BRKGA, Este parâmetro é um simples booleano, onde verdade indica que a decodificação será efetuada em GPU, e falso indica que a decodificação ocorrerá em CPU.

Init: É o método que é chamado na inicialização do algoritmo, e sua finalidade é garantir ao usuário a possibilidade de inicializar possíveis instâncias que o decodificador poderá utilizar.

Um exemplo para um decodificador para a função Rastrigin foi implementado, pode ser encontrado pelo nome SampleDecoder.cu.

Uma vez especificada a classe decoder, o usuário prosseguirá com a implementação do arquivo que servirá como base para resolver o problema escolhido., um exemplo para este arquivo está contido no repositório, você poderá encontrá-lo pelo nome api-usage.cu.

Para este arquivo, alguns métodos devem ser chamados:

Inicializar o GPU-BRKGA a partir do contrutor da classe.

Após isso, o usuário deve chamar a função evolve, a função evolve pode ser chamada passando um parâmetro que indica a quantidade de gerações que as populações irão evoluir, por exemplo: `evolve(1000)`, assim, evoluindo por mil gerações, ou então chamar simplesmente `evolve()`, que irá evoluir por uma geração e assim controlar a quantidade de gerações por meio de um laço.

Então, após as gerações desejadas, o usuário pode obter a melhor solução encontrada por meio do método `getBestIndividual()`, que irá retornar a melhor solução.

Podem ainda ser utilizados outros métodos presentes em nossa API, como por exemplo: `reset()`, que irá reinciar todas as populações do algoritmo com novas chaves aleatórias, `exchangeElite(M)`, que irá trocar os M indivíduos elite entre as populações independentes, caso utilize mais de uma população, `getPopulations()` que irá retornar todas as populações presentes no algoritmo, `cpyHost()`, que copia todos os dados em memória VRAM para a RAM, além dos já conhecidos métodos get para acessar valores privados ao GPU-BRKGA.









