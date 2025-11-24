## Trabalho_Final_RL

 # Rodando o algoritmo inicial :
  Para rodar o código original, instalamos todas as dependências usando WSL (para que o sistema tivesse suporte em Linux, já que o Windows teve problema em aceitar algumas dependências). Para que ele rodasse, fizemos apenas uma alteração na função "create_population()", devido a versão do AgilRL usada. O código pode ser visto em Original.ipynb, e demorou 3h e 15 min para rodar totalmente. O resultado obtido da pontuação média da população de acordo com a iteração pode ser vista no gráfico:  
  
  
<img width="886" height="439" alt="image" src="https://github.com/user-attachments/assets/daef97e3-0df0-4027-811e-9f559e20bffa" />


  Analisamos que o algoritmo incial apresenta uma instabilidade muito grande no treinamento, oscilando em sua pontuação média e não tendo uma melhoria estável na sua performence ao longo do tempo. Para tentar superar esse fato, inicialmente mudamos alguns hiperparâmetros, de acordo com as seguintes justificativas: 

## 1) Mudança nos parâmetros de mutação: 

  Consideramos essa a alteração mais importante nos parâmetros, pois entedemos que os parâmetros de mutação estavam muito altos. Isso explicaria a oscilação da pontuação média grande, e a dificuldade dos agentes de aprender, pois os agentes estão sendo alterados com muita frequência e com muita intensidade. Por consequência, mudamos os seguintes parâmetros: 

  - Aumentamos muito a chance das mutações não acontecerem.
  - Diminuimos a chance das mudanças na arquitetura.
  - Diminuimos muito a chance de adcionar outra camada.
  - Diminuimos muito a chance de mudança dos parâmetros, para manter o aprendizado feito pelos agentes.
  - Diminuimos a intensidade de cada mutação.

Apesar dessas mudanças, não alteramos a probabilidade dos hiper-parâmetros de RL mudarem (dado que o agente passou por uma mutação, o que tornamos mais improvável), pois consideramos importante os agentes experimentarem o aprendizado com diferentes hiper-parâmetros. Porém, fizemos algumas alterações nas faixas aceitáveis para os hiper-parâmetros.

## 2) Mudança na faixa de hiperparâmetros. 

  Consideramos que a faixa permitida de hiper-parâmetros de aprendizado era muito larga, e incentivava muito a exploração de valores extremos que claramente não iriam ser ótimos. Por consequência, fizemos as seguintes alterações: 

  - Diminuímos muito o máximo do learning rate do ator.
  - Diminuímos o máximo do learning rate do critic (mas deixamos maior que o do ator).
  - Aumentamos muito o mínimo do Batch Size.
  - Diminuimos o máximo do Batch Size.

Apesar disso, mantivemos o Learn Step mais ou menos na mesma faixa, pois achamos importante explorar as diferentes possibilidades para esse hiper-parâmetro.

## 3) Mudança nos hiper-parâmetros iniciais. 

  Ajustamos também alguns hiper-parâmetros iniciais para valores que consideramos mais adequados, mas consideramos essas alterações menos importantes: 

- Aumentamos o Batch Size inicial.
- Diminuimos a escala do ruído (devido ao número de ambientes).
- Aumentamos o Learning Rate do ator.
- Diminuimos o Learn Step.
- Diminuimos Tau
- Aumentamos Gamma consideravelmente.

O resultado de todas essas mudanças foi o seguinte: 


<img width="1190" height="590" alt="image" src="https://github.com/user-attachments/assets/466a8882-2889-4a2f-bcfa-2d6b5c072bd1" />




