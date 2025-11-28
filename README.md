Trabalho Final RL:  

  
Francisco Kuchpil e Heitor Trielli 

 # Rodando o algoritmo inicial :
  Para rodar o código original, instalamos todas as dependências usando WSL (para que o sistema tivesse suporte em Linux, já que o Windows teve problemas em aceitar algumas dependências). Para que ele rodasse, fizemos apenas uma alteração na função "create_population()", devido a versão do AgilRL usada. O código pode ser visto em Original.ipynb, e demorou 3h e 15 min para rodar totalmente. O resultado obtido da pontuação média da população de acordo com a iteração pode ser visto no gráfico:  
  
  
<img width="886" height="439" alt="image" src="https://github.com/user-attachments/assets/daef97e3-0df0-4027-811e-9f559e20bffa" />
  
Analisamos que o algoritmo inicial apresenta uma instabilidade muito grande no treinamento, oscilando em sua pontuação média e não tendo uma melhoria estável na sua performence ao longo do tempo. 

# Rodando o algoritmo com mudança de parâmetros:  

Para tentar superar a instabilidade da performance dos agentes, as primeiras alterações que fizemos no algoritmo foi a mudança de alguns parâmetros. Fizemos isso de acordo com as seguintes justificativas:  

## 1) Mudança nos parâmetros de mutação: 

  Consideramos essa a alteração mais importante, pois entendemos que os parâmetros de mutação estavam muito altos. Isso explicaria a oscilação da pontuação média grande, e a dificuldade dos agentes de aprender, pois os agentes estão sendo alterados com muita frequência e com muita intensidade. Por consequência, fizemos as seguintes adaptações: 

  - Aumentamos muito a chance das mutações não acontecerem.
  - Diminuimos a chance das mudanças na arquitetura.
  - Diminuimos muito a chance de adcionar outra camada.
  - Diminuimos muito a chance de mudança dos parâmetros, para manter o aprendizado feito pelos agentes.
  - Diminuimos a intensidade de cada mutação.

Apesar dessas mudanças, não alteramos a probabilidade dos hiper-parâmetros de RL mudarem (dado que o agente passou por uma mutação, o que tornamos mais improvável), pois consideramos importante os agentes experimentarem o aprendizado com diferentes hiper-parâmetros. Porém, fizemos algumas alterações nas faixas aceitáveis para tais hiper-parâmetros.

## 2) Mudança na faixa de hiperparâmetros. 

  Consideramos que a faixa permitida de hiper-parâmetros de aprendizado era muito larga, e incentivava muito a exploração de valores extremos que não achamos que seriam ótimos. Por consequência, fizemos as seguintes alterações: 

  - Diminuímos muito o máximo do learning rate do ator.
  - Diminuímos o máximo do learning rate do critic (mas deixamos maior que o do ator).
  - Aumentamos muito o mínimo do Batch Size.
  - Diminuimos o máximo do Batch Size.

Apesar disso, mantivemos as possibilidades de Learn Step mais ou menos na mesma faixa, pois achamos importante explorar as diferentes possibilidades para esse hiper-parâmetro.

## 3) Mudança nos hiper-parâmetros iniciais. 

  Ajustamos também alguns hiper-parâmetros iniciais para valores que consideramos mais adequados: 

- Aumentamos o Batch Size inicial.
- Diminuimos a escala do ruído (devido ao grande número de ambientes).
- Aumentamos o Learning Rate do ator.
- Diminuimos o Learn Step.
- Diminuimos Tau.
- Aumentamos Gamma consideravelmente.

O código rodado pode ser visto em Parametros.ipnyb, e resultado de todas essas mudanças foi o seguinte: 


<img width="1190" height="590" alt="image" src="https://github.com/user-attachments/assets/466a8882-2889-4a2f-bcfa-2d6b5c072bd1" />


É possível ver que as mudanças deixaram o algoritmo mais estável, além de melhorar o desempenho. Essa melhora na performance inspirou outras mudanças no algoritmo:


# Diminuindo os paramêtros de mutação ao longo do treino:  

Interpretamos que diminuir os parâmetros de mutação foi muito positivo para a performance dos agente ao longo do tempo. Nas novas alterações no código diminuimos ainda mais a probabilidade de novas camadas nas mutações, e zeramos a probabilidade de mutações na arquitetura, variáveis que consideramos muito destrutivas e com pouco retorno pela exploração (depois, percebemos que não era bem assim). Porém, consideramos que diminuir muito as outras probabilidades de mutação dos agentes reduziria demais a exploração, e poderíamos ficar presos a agentes com parâmetros ruins. Portanto, decidimos diminui-las ao longo do tempo, favorecendo assim uma maior exploração no ínicio e uma maior exploitação no final. Para implementar essa mudança, definimos a variável progress, que é uma fração do maior número de passos dado por um agente (variável que controla a continuidade do loop de treinamento) pelo número máximo de passos. Assim, 0 ≤ progresso < 1.

Depois, criamos a variável decay, que é igual a (1 - 0.9 * progress), ou seja, varia de 1 a 0.1 conforme vamos avançando no treinamento. Multiplicamos todas os parâmetros de mutação por elas, ou seja, diminuimos progressivamente a probabilidade de cada mutação ao longo do treino (além da força das mutações).

Além disso, também aumentamos progressivamente a probabilidade de não haver uma mutação nos agentes. Estabelecendo base como a probabilidade inicial de não haver uma mutação, definimos a probabilidade de não haver uma mutação em um determinado momento do treino como base + (1 - base) * progress.


Também aumentamos consideravelmente (de 1 para 5) a variável eval_loop. Ela controla quantos valores de fitness são usados para avaliar os agentes, e aumentar ela diminui o "azar" de modelos ruins ganharem de bons. Assim, esperamos também estabilizar o aprendizado. 

O código rodado pode ser visto em Diminuicao.ipnyb, e resultado dessas mudanças foi o seguinte:

<img width="886" height="439" alt="image" src="https://github.com/user-attachments/assets/0da5b830-70c0-4a03-9e09-caa138493601" />


Analisamos, rodando o código várias vezes, que o agente teve uma ótima curva de aprendizado, mas teve quedas abruptas e dificuldade em manter os melhores modelos. 

# Treinando em três fases :

Percebemos que o problema tratado é extremamente instável. Pela gráfico das pontuações médias, nossa configuração era capaz de alcançar um bom desempenho, com a exploração de parâmetros sendo feita pela mutação e seleção de bons agentes. Porém, a população tem muitas quedas abruptas em sua performance, e não consegue estabilizar bons resultados.
  
Portanto, decidimos dividir o treinamento dos agentes em três fases. Na primeira, de 0 a 40 por cento do treinamento, treinamos os agentes da mesma forma que treinamos os agentes anteriores. Definimos os praticamente os mesmos padrões de mutação, e o mesmo termo de decaimento para esses parâmetros. A única diferença é que voltamos a colocar as probabilidades de mutação na arquitetura e new layer (ambas como 0.05), um valor ainda baixo. Fizemos isso porque achamos que na fase inicial essas mutações não seriam muito destrutivas, pois o modelo ainda estaria sendo treinado, e vale a pena explorar essas mutações.
  
Na segunda fase (entre 40 a 80 por cento do treinamento), fizemos duas alterações: A primeira foi diminuir a probabilidade dessas duas mutações para 0.01. Consideramos que com o modelo já mais treinado, essas mutações passam a ser muito destrutivas e vale menos a pena explorar elas. A segunda mudança foi parar totalmente de mutar o melhor agente da população. Assim, a população continuaria passando por mutações, mas teremos a vantagem de continuar sempre com o melhor agente após as avaliações. Isso diminuirá a exploração (pois teremos 3 possibilidades de mutação após mutação ao invés de duas), mas deve garantir uma estabilidade muito maior para a performance dos agentes em uma fase em que eles já passaram por um treinamento com mais exploração.

Na terceira (e última) fase, decidimos parar totalmente com as mutações. Entendemos que essa fase é própria apenas para ajuste dos agentes, e forçar mudanças neles deixa de fazer sentido. Nessa fase os agentes ainda vão passar por seleção entre eles e aprendizado, mas não estaremos mais mudando seus parâmetros. Esperamos assim uma estabilidade muito grande da população, e uma melhora bem suave no desempenho dos agentes.

O código pode ser visto em "Fases.ipynb". Esses foram o resultados:

<img width="1190" height="590" alt="image" src="https://github.com/user-attachments/assets/72cd7c74-e1fc-4978-aeff-3b9f16b6b692" />

Rodamos o código e analisamos que os agentes conseguiram aprender bem, chegando a ter médias de -20 em suas performances. Entretanto, elas decaíram em um momento que projetamos maior estabilidade. 

Porém, tentando rodar o código mais vezes, percebemos que essa primeira curva foi "sortuda". Não conseguimos reproduzir resultados tão bons mesmo sem alterações no código, e tivemos que alterar alguns parâmtros novamente. 

# Alteração final: 

Nosso modelo deixava seus parâmetros de aprendizado muito a mercê das mutações aleatórias, e por isso rodar o mesmo código várias vezes resultava em performances muito diversas. Portanto, foi necessário estreitar ainda mais as faixas de parâmetros que nosso modelo poderia assumir após as mutações. Refinamos os parâmetros que tinhamos definido testando o aprendizado de apenas um agente sem mutações. Fizemos as seguintes mudanças: 

- Diminuimos o LR inicial do actor e do critic (e igualamos eles)
- Estreitamos a faixa de valores possíveis de LR após as mutações
- Diminuimos o learn step inicial
- Estreitamos a faixa aceitável para o learn step após as mutações
- Estreitamos a faixa aceitável de batch size

Também diminuimos a probabilidade de mutação nos parâmetros, e zeramos a probabilidade da mutação new layer. Assim nosso código teria muito mais estabilidade, e os torneios seriam mais para comparar agentes com graus de aprendizado diferente do que explorar diferentes parâmetros (apesar dessa exploração ainda existir em uma faixa muito mais estreita).

Para além disso, mantivemos as mesmas três fases para o treinamento. Na primeira mutamos toda a população após os torneios, na segunda não mutamos o melhor agente e na terceira paramos totalmente as mutações, e só realizamos torneios. Porém, na terceira fase adicionamos um termo de decaimento para os learning rates dos agentes, que passa a ser 0.9 vezes o valor antigo toda vez que fazemos um torneio, com um limite inferior de 1^e-6. Tanto o lr do actor quando o lr do critic passam por essa mudança. Assim, esperamos fazer um fine tuning mais gradual, sem grandes alterações no comportamento dos agentes devido a learning rates altos. 

Portanto teremos as três fases: Exploração controlada de parâmetros, refinamento com mutações fracas e fine tuning com LR decaindo. 

O código pode ser visto em "Final.ipynb". Esses foram o resultados:

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/315a32cf-15f7-45d5-b269-81f2ce02ab37" />

Consideramos os agentes tiveram performances boas e estáveis, justificando todas as alterações que fizemos no código.








