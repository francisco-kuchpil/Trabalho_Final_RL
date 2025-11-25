Trabalho Final RL:  

  
Francisco Kuchpil e Heitor Trielli 

 # Rodando o algoritmo inicial :
  Para rodar o código original, instalamos todas as dependências usando WSL (para que o sistema tivesse suporte em Linux, já que o Windows teve problemas em aceitar algumas dependências). Para que ele rodasse, fizemos apenas uma alteração na função "create_population()", devido a versão do AgilRL usada. O código pode ser visto em Original.ipynb, e demorou 3h e 15 min para rodar totalmente. O resultado obtido da pontuação média da população de acordo com a iteração pode ser vista no gráfico:  
  
  
<img width="886" height="439" alt="image" src="https://github.com/user-attachments/assets/daef97e3-0df0-4027-811e-9f559e20bffa" />


  Analisamos que o algoritmo incial apresenta uma instabilidade muito grande no treinamento, oscilando em sua pontuação média e não tendo uma melhoria estável na sua performence ao longo do tempo. 

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

Interpretamos que diminuir os parâmetros de mutação foi muito positivo para a performance dos agente ao longo do tempo. Nas novas alterações no código diminuimos ainda mais a probabilidade de novas camadas nas mutações, e zeramos a probabilidade de mutações na arquitetura, variáveis que consideramos muito destrutivas e com pouco retorno pela exploração (depois, percebemos que não era bem assim). Porém, consideramos que diminuir muito as outras probabilidades de mutação dos agentes reduzeria demais a exploração, e poderíamos ficar presos a agentes com parâmetros ruins. Portanto, decidimos diminui-las ao longo do tempo, favorecendo assim uma maior exploração no ínicio e uma maior exploitação no final. Para implementar essa mudança, definimos a variável progress, que é uma fração do maior número de passos dado por um agente (variável que controla a continuidade do loop de treinamento) pelo número máximo de passos. Assim, 0 ≤ progresso < 1.

Depois, criamos a variável decay, que é igual a (1 - 0.9 * progress), ou seja, varia de 1 a 0.1 conforme vamos avançando no treinamento. Multiplicamos todas os parâmetros de mutação por elas, ou seja, diminuimos progressivamente a probabilidade de cada mutação ao longo do treino.

Além disso, também aumentamos progressivamente a probabilidade de não haver uma mutação nos agentes. Estabelecendo base como a probabilidade inicial de não haver uma mutação, definimos a probabilidade de não haver uma mutação em uma determinada momento do treino como base + (1 - base) * progress.


Também aumentamos consideravelmente (de 1 para 5) a variável eval_loop. Ela controla quantos valores de fitness são usados para avaliar os agentes, e aumentar ela diminui o "azar" de modelos ruins ganharem de bons. Assim, esperamos também estabilizar o aprendizado. 

O código rodado pode ser visto em Diminuicao.ipnyb, e resultado dessas mudanças foi o seguinte:

<img width="886" height="439" alt="image" src="https://github.com/user-attachments/assets/0da5b830-70c0-4a03-9e09-caa138493601" />


Analisamos, rodando o código várias vezes, que o agente teve uma ótima curva de aprendizado, mas teve quedas abruptas e dificuldade em manter os melhores modelos. Essa análise inspirou a última mudança que fizemos no código:

# Treinando em três fases :

Percebemos que o problema tratado é extremamente instável. Pela gráfico das pontuações médias, nossa configuração era capaz de alcançar um bom desempenho, com a exploração de parâmetros sendo feita pela mutação e seleção de bons agentes. Porém por se tratar de quatro agentes interagindo em um mesmo ambiente, sabemos que a mutação em um deles altera a performance de todos. Se temos um bom desempenho dos agentes, temos que eles estão tomando decisões em um "espaço estreito", e uma alteração em seu ambiente prejudica significativamente seu desempenho e seu aprendizado. Portanto, interpretamos da curva que os agentes estavam bons, e uma sequência de mutações ruins atrapalharam seu desempenho, e a instabilidade também no ambiente causadas pelas mutações fez com que os agentes bons inicialmente tivessem menos chance de serem selecionados. É um problema com quatro agentes (pouco) interagindo em um mesmo ambiente, com mutações e seleções de agente frequentes, o que faz com que a instabilidade na população seja muito grande.
  
Portanto, decidimos dividir o treinamento dos agentes em três fases. Na primeira, de 0 a 40 por cento do treinamento, treinamos os agentes da mesma forma que treinamos os agentes anteriores. Definimos os praticamente os mesmos padrões de mutação, e o mesmo termo de decaimento para esses parâmetros. A única diferença é que voltamos a colocar as probabilidades de mutação na arquitetura e new layer (ambas como 0.05), um valor ainda baixo. Fizemos isso porque achamos que na fase inicial essas mutações não seriam muito destrutivas, pois o modelo ainda estaria sendo treinado, e vale a pena explorar essas mutações.
  
Na segunda fase (entre 40 a 80 por cento do treinamento), fizemos duas alterações: A primeira foi diminuir a probabilidade dessas duas mutações para 0.01. Consideramos que com o modelo já mais treinado, essas mutações passam a ser muito destrutivas e vale menos a pena explorar elas. A segunda mudança foi parar totalmente de mutar o melhor agente da população. Assim, a população continuaria passando por mutações, e muitas vezes o própio melhor agente teria cópias mutadas adcionadas na população (caso ele fosse selecionado em um sorteio), permitindo assim também a exploração em torno de seus parâmetros. Assim, teríamos a vantagem de continuar sempre com o melhor agente após as avaliações. Isso diminuirá a exploração (pois teremos 3 possibilidades de mutação após mutação ao invez de duas), mas deve garantir uma estabilidade muito maior para a performance dos agentes na fase em que eles já estão treinados.

Na terceira (e última) fase, decidimos parar totalmente com as mutações. Entendemos que essa fase é própria apenas para ajuste dos agentes, e forçar mudanças neles deixa de fazer sentido. Nessa fase os agentes ainda vão passar por seleção entre eles e aprendizado, mas não estaremos mais mudando seus parâmetros. Esperamos assim uma estabilidade muito grande da população, e uma melhora bem suave no desempenho dos agentes.

Essa foi a última alteração que fizemos no código, e ele pode ser visto em "Fases.ipynb". Esses foram o resultados:



