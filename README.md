# Trabalho Final RL
### Francisco Kuchpil e Heitor Trielli 

## Rodando o algoritmo inicial
Para rodar o código original, instalamos todas as dependências usando WSL (para que o sistema tivesse suporte em Linux, já que o Windows teve problemas em aceitar algumas dependências). Para que ele rodasse, fizemos apenas uma alteração na função "create_population()", devido a versão do AgileRL usada. O código pode ser visto em Original.ipynb, e demorou 3h e 15 min para rodar totalmente. O resultado obtido da pontuação média da população de acordo com a iteração pode ser visto no gráfico abaixo:  

  
<img width="886" height="439" alt="image" src="https://github.com/user-attachments/assets/daef97e3-0df0-4027-811e-9f559e20bffa" />
  
Podemos verificar que que o algoritmo inicial apresenta uma instabilidade muito grande no treinamento, oscilando em sua pontuação média e não tendo uma melhoria estável na sua performence ao longo do tempo. 

## Rodando o algoritmo com mudança de parâmetros  

Para tentar superar a instabilidade da performance dos agentes, a primeira alteração que fizemos no algoritmo foi a mudança de alguns parâmetros. Fizemos isso de acordo com as seguintes justificativas:  

#### 1) Mudança nos parâmetros de mutação 

  Consideramos essa a alteração mais importante, pois entendemos que os parâmetros de mutação estavam muito altos. Isso explicaria a oscilação da pontuação média grande, e a dificuldade dos agentes de aprender, tendo em vista que os agentes estão sendo alterados com muita frequência e intensidade. Desta forma, fizemos as seguintes adaptações: 

  - Diminuimos consideravelmente a chance das mutações acontecerem.
  - Diminuimos a chance das mudanças na arquitetura.
  - Diminuimos consideravelmente a chance de adicionar outra camada.
  - Diminuimos consideravelmente a chance de mudança dos parâmetros, para manter o aprendizado feito pelos agentes.
  - Diminuimos a intensidade de cada mutação.

Apesar dessas mudanças, não alteramos a probabilidade dos hiper-parâmetros de RL mudarem dado que o agente passou por uma mutação, pois consideramos importante os agentes experimentarem o aprendizado com diferentes hiper-parâmetros. Fizemos, porém, algumas alterações nas faixas aceitáveis para tais hiper-parâmetros, descritas abaixo.

#### 2) Mudança na faixa de hiperparâmetros

  Consideramos que a faixa permitida de hiper-parâmetros de aprendizado era muito larga, e incentivava a exploração de valores extremos que não seriam ótimos. Por consequência, fizemos as seguintes alterações: 

  - Diminuímos consideravelmente o máximo do learning rate do ator.
  - Diminuímos o máximo do learning rate do critic, mas ainda deixamos maior que o do ator.
  - Aumentamos consideravelmente o mínimo do Batch Size.
  - Diminuimos o máximo do Batch Size.

Mantivemos, porém, as possibilidades de Learn Step na mesma faixa, pois achamos importante explorar as diferentes possibilidades para esse hiper-parâmetro.

#### 3) Mudança nos hiper-parâmetros iniciais

  Ajustamos também alguns hiper-parâmetros iniciais para valores que consideramos mais adequados: 

- Aumentamos o Batch Size inicial.
- Diminuimos a escala do ruído, devido ao grande número de ambientes.
- Aumentamos o Learning Rate do ator.
- Diminuimos o Learn Step.
- Diminuimos Tau.
- Aumentamos Gamma consideravelmente.

O código rodado pode ser visto em "Parametros.ipnyb", e o resultado de todas essas mudanças pode ser visto no gráfico abaixo. 


<img width="1190" height="590" alt="image" src="https://github.com/user-attachments/assets/466a8882-2889-4a2f-bcfa-2d6b5c072bd1" />


É possível ver que as mudanças deixaram o algoritmo mais estável, além de melhorar o desempenho.


## Diminuindo os parâmetros de mutação ao longo do treino  

Interpretamos que diminuir os parâmetros de mutação foi muito positivo para a performance dos agente ao longo do tempo, mas conseguimos melhorar ainda mais o desempenho do modelo. Nas novas alterações no código diminuimos ainda mais a probabilidade de novas camadas nas mutações, e zeramos a probabilidade de mutações na arquitetura, variáveis que consideramos muito destrutivas e com pouco retorno pela exploração. Porém, consideramos que diminuir muito as outras probabilidades de mutação dos agentes reduziria demais a exploração, e poderíamos ficar presos a agentes com comportamentos ruins. Portanto, decidimos diminui-las ao longo do tempo, favorecendo assim uma maior exploração no ínicio e uma maior exploitação no final. Para implementar essa mudança, definimos a variável progress, que é uma fração do maior número de passos dado por um agente (variável que controla a continuidade do loop de treinamento) pelo número máximo de passos. Assim, 0 ≤ progresso < 1.

Depois, criamos a variável decay, que é igual a 1 - 0.9 * progress, de modo que ela varia de 1 a 0.1 conforme vamos avançando no treinamento. Multiplicamos todas os parâmetros de mutação por elas. Ou seja, diminuimos progressivamente a probabilidade de cada mutação ao longo do treino, além da força das mutações.

Além disso, também aumentamos progressivamente a probabilidade de não haver uma mutação nos agentes. Estabelecendo base como a probabilidade inicial de não haver uma mutação, definimos a probabilidade de não haver uma mutação em um determinado momento do treino como base + (1 - base) * progress.

Também aumentamos consideravelmente (de 1 para 5) a variável eval_loop. Ela controla quantos valores de fitness são usados para avaliar os agentes, e aumentar ela diminui o "azar" de modelos ruins ganharem de bons. Assim, esperamos também estabilizar o aprendizado. 

O código rodado pode ser visto em "Diminuicao.ipnyb", e resultado dessas mudanças pode ser visto no gráfico abaixo.

<img width="886" height="439" alt="image" src="https://github.com/user-attachments/assets/0da5b830-70c0-4a03-9e09-caa138493601" />


Analisamos, ao rodar o código várias vezes, que os agentes até conseguem chegar a bons resultados, mas apresentam quedas abruptas e dificuldade em manter os melhores modelos. 

## Treinando em três fases

Percebemos que o problema tratado é instável. Pela gráfico das pontuações médias, nossa configuração era capaz de alcançar um bom desempenho, com a exploração de parâmetros sendo feita pela mutação e seleção de bons agentes. Porém, a população tem muitas quedas abruptas em sua performance, e não consegue estabilizar bons resultados.
  
Portanto, decidimos dividir o treinamento dos agentes em três fases. Na primeira, de 0 a 40 por cento do treinamento, treinamos os agentes da mesma forma que treinamos os agentes anteriores. Definimos os mesmos padrões de mutação, e o mesmo termo de decaimento para esses parâmetros. A única diferença é que voltamos a colocar as probabilidades de mutação na arquitetura e new layer ambas como 0.05, um valor ainda baixo. Fizemos isso porque achamos que na fase inicial essas mutações não seriam muito destrutivas, dado que o modelo ainda estaria sendo treinado.
  
Na segunda fase (entre 40 a 80 por cento do treinamento), fizemos duas alterações: a primeira foi diminuir a probabilidade dessas duas mutações para 0.01. Consideramos que com o modelo já tendo explorado, essas mutações passam a ser mais destrutivas que positivas, e a exploração passa a valer menos a pena. A segunda mudança foi parar totalmente de mutar o melhor agente da população. Assim, a população continuaria passando por mutações, mas teremos a vantagem de continuar sempre com o melhor agente após as avaliações. Isso diminuirá a exploração, pois teremos três possibilidades de mutação após os torneios ao invés de quatro, mas deve garantir uma estabilidade maior para a performance dos agentes em uma fase em que eles já passaram por um treinamento com mais exploração.

Na terceira e última fase, decidimos parar totalmente com as mutações. Entendemos que essa fase é própria apenas para ajuste dos agentes pelo seus próprios aprendizados, e forçar mudanças neles deixa de fazer sentido. Nessa fase os agentes ainda vão passar por seleção entre eles e aprendizado, mas não estaremos mais mudando seus parâmetros. Esperamos assim uma estabilidade maior da população, e uma melhora mais suave no desempenho dos agentes.

O código pode ser visto em "Fases.ipynb". Esses foram o resultados:

<img width="1190" height="590" alt="image" src="https://github.com/user-attachments/assets/72cd7c74-e1fc-4978-aeff-3b9f16b6b692" />

Rodamos o código e analisamos que os agentes conseguiram aprender bem, chegando a ter médias de -20 em suas performances. Entretanto, elas decaíram em um momento que projetamos maior estabilidade. 

Porém, ao executar o código mais vezes, percebemos que essa primeira curva foi "sortuda". Nas demais execuções, mesmo sem alterar o código, os resultados foram piores, o que nos levou a alteração final, descrita na próxima sessão. 

## Alteração final

Nosso modelo deixava seus parâmetros de aprendizado muito 'à mercê' das mutações aleatórias, e por isso rodar o mesmo código alterando a seed resultava em performances muito diversas. Portanto, foi necessário estreitar ainda mais as faixas de parâmetros que nosso modelo poderia assumir após as mutações. Refinamos então os parâmetros que tinhamos definido testando o aprendizado de apenas um agente sem mutações. Fizemos as seguintes mudanças: 

- Diminuimos o learning rate inicial do actor e do critic (e igualamos eles)
- Estreitamos a faixa de valores possíveis de learning rate após as mutações
- Diminuimos o learn step inicial
- Estreitamos a faixa aceitável para o learn step após as mutações
- Estreitamos a faixa aceitável de batch size

Também diminuimos a probabilidade de mutação nos parâmetros, e zeramos a probabilidade da mutação new layer. Assim nosso modelo terá mais estabilidade, e os torneios serão mais focados em comparar agentes com graus de aprendizado diferente do que explorar diferentes parâmetros, apesar dessa exploração ainda existir em uma faixa mais estreita.

Para além disso, mantivemos as mesmas três fases para o treinamento. Na primeira mutamos toda a população após os torneios, na segunda não mutamos o melhor agente e na terceira paramos totalmente as mutações, apenas realizando os torneios. Porém, na terceira fase adicionamos um termo de decaimento para os learning rates dos agentes, que passa a ser 0.9 vezes o valor antigo toda vez que fazemos um torneio, com um limite inferior de 1^e-6. Tanto o learning rate do actor quando o learning rate do critic passam por essa mudança. Assim, esperamos fazer um fine tuning mais gradual, sem grandes alterações no comportamento dos agentes devido a learning rates altos. 

Portanto teremos as três fases: exploração controlada de parâmetros, refinamento com mutações fracas e fine tuning sem mutações com learning rate decaindo. Continuamos com o decaimento graudal da probabilidade de mutações na fase 1 e 2, como na última versão do código. 

O código pode ser visto em "Final.ipynb". Os resultados podem ser vistos no gráfico abaixo.

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/315a32cf-15f7-45d5-b269-81f2ce02ab37" />

Consideramos que os agentes tiveram performances boas e estáveis, justificando as alterações que fizemos no código. Essa última versão do código também está disponível em "Alterado.py".








