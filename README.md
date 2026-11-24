Sistema de Monitoramento de Sensores
Matéria de Análise de sensores com Streamlit e IA
1. Apresentação do projeto

Este projeto foi desenvolvido como atividade acadêmica de Inteligência Artificial, com o objetivo de analisar dados de sensores e prever possíveis falhas em equipamentos.

A aplicação utiliza dados de temperatura, umidade e vibração para identificar situações de funcionamento normal ou possíveis falhas.

2. Objetivo

Desenvolver um sistema que auxilie no monitoramento de equipamentos, utilizando técnicas de aprendizado de máquina para identificar padrões associados a falhas.

3. Tecnologias utilizadas
Python 3.12
Streamlit
Pandas
Scikit-learn
Plotly
NumPy
uv
Git e GitHub
4. Base de dados

Foi utilizado o arquivo data/sensores.csv, disponibilizado no template da atividade.

A base contém 800 leituras simuladas e as seguintes colunas:

timestamp: data e horário da leitura.
temperatura: temperatura registrada.
umidade: umidade registrada.
vibracao: nível de vibração.
falha: classificação da leitura, sendo 0 para funcionamento normal e 1 para falha.
5. Funcionalidades da aplicação

A aplicação possui quatro abas:

5.1 Dados dos sensores

Apresenta os registros da base de dados em uma tabela, gráficos de evolução dos sensores e distribuição das leituras normais e com falha.

5.2 Modelo de IA

Utiliza o algoritmo Random Forest Classifier para prever falhas com base nas variáveis temperatura, umidade e vibração.

Os dados são divididos em 80% para treinamento e 20% para teste.

As métricas utilizadas são:

Acurácia: percentual geral de previsões corretas.
Precisão: proporção de previsões de falha que realmente eram falhas.
Recall: proporção de falhas reais identificadas pelo modelo.
F1-score: equilíbrio entre precisão e recall.
5.3 Nova leitura

Permite informar novos valores de temperatura, umidade e vibração para obter uma previsão do modelo.

A aplicação apresenta a classificação, a probabilidade estimada de falha e uma sugestão de ação: monitorar, inspecionar ou parar e avaliar o equipamento.

5.4 Limitações e cuidados

Apresenta riscos e limitações do modelo, além de medidas para reduzir esses problemas.

6. Limitações do projeto e formas de redução dos riscos
Dados simulados: podem não representar completamente equipamentos reais. Para reduzir esse risco, é necessário validar o modelo com dados reais.
Previsões incorretas: o modelo pode deixar de identificar uma falha ou indicar uma falha inexistente. Para reduzir esse risco, devem ser avaliadas as métricas, principalmente o recall, e realizadas inspeções técnicas.
Poucas variáveis: são utilizados apenas temperatura, umidade e vibração. Para reduzir esse risco, podem ser incluídos outros sensores, como pressão e ruído.
Mudanças nas condições de operação: podem prejudicar o desempenho do modelo. Para reduzir esse risco, é importante acompanhar os resultados e atualizar o treinamento com novos dados.

As previsões devem servir como apoio à manutenção, nunca como substituição da avaliação de profissionais responsáveis pela segurança dos equipamentos.

7. Como executar o projeto

É necessário ter Python e uv instalados no computador.

Abra o terminal dentro da pasta do projeto e execute:

powershell
uv python install 3.12
uv sync
uv run streamlit run app.py


para facilitar, o projeto pode ser aberto no navegador:

https://monitoramento-sensores-svqnztdyruuurrwdm4ju5a.streamlit.app/

8. Conclusão

O projeto demonstra como a Inteligência Artificial pode auxiliar na manutenção preventiva de equipamentos por meio da análise de dados de sensores.

Apesar de apresentar resultados úteis, o modelo possui limitações e precisa ser validado com dados reais antes de qualquer utilização em situações operacionais.

Projeto acadêmico de monitoramento de sensores e previsão de falhas.
