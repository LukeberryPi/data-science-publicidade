from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl

# importando a base de dados
df = pd.read_csv(r"Intensivão\aula4 - Data Science\advertising.csv")
# print(df)

# descobrir a correlação entre as diferentes informações
# por meio de gráficos

# criar o gráfico
sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)

# exibir o gráfico
plt.show()

# separar os dados em X e Y
y = df["Vendas"]
x = df[["TV", "Radio", "Jornal"]]

# separar os dados em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.3, random_state=1)

# treina no simulado
# teste na prova para ver se aprendeu

# criando a IA
# importar

# cria a inteligência
modelo_regressao_linear = LinearRegression()
modelo_arvore_decisao = RandomForestRegressor()

# treina a inteligência
modelo_regressao_linear.fit(x_treino, y_treino)
modelo_arvore_decisao.fit(x_treino, y_treino)

# Teste da IA e Avaliação do Melhor Modelo
# cria as previsoes

previsao_regressão_linear = modelo_regressao_linear.predict(x_teste)
previsao_arvore_decisao = modelo_arvore_decisao.predict(x_teste)

# compara as previsões com o gabarito
print(metrics.r2_score(y_teste, previsao_regressão_linear))
print(metrics.r2_score(y_teste, previsao_arvore_decisao))

# o MELHOR modelo é o modelo de Árvore de Decisão
# como esse é o melhor modelo, usamos para fazer novas previsões
novos_valores = pd.read_csv(r"Intensivão\aula4 - Data Science\novos.csv")
print(novos_valores)

nova_previsao = modelo_arvore_decisao.predict(novos_valores)
print(nova_previsao)
