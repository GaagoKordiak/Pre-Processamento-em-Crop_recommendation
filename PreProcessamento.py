import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados
# O arquivo "Crop_recommendation.csv" esta no mesmo diretório
df = pd.read_csv('Crop_recommendation.csv')

# Exibir as primeiras linhas e informações básicas do dataset
print("--- Informações Iniciais do Dataset ---")
print(df.head())
print("\n--- Informações de Colunas e Tipos de Dados ---")
print(df.info())
print("\n--- Estatísticas Descritivas ---")
print(df.describe())

# 2. Tratamento de valores ausentes (se houver)
# Verificando a quantidade de valores nulos
print("\n--- Verificando valores ausentes ---")
print(df.isnull().sum())
# Neste dataset não há valores ausentes.
# Se houvesse, poderíamos usar df.fillna(df.mean()) para preenchê-los com a média, por exemplo.

# 3. Criação de novas variáveis derivadas das originais
# 'índice de umidade_temperatura': A relação entre umidade e temperatura pode ser um bom indicador.
# 'índice_NPK': Uma média ponderada dos principais nutrientes (Nitrogênio, Fósforo, Potássio).
print("\n--- Criando novas variáveis ---")
df['humidity_temp_ratio'] = df['humidity'] / df['temperature']
df['NPK_index'] = (df['N'] + df['P'] + df['K']) / 3

print(df.head())

# 4. Codificação de variáveis categóricas
# Usando LabelEncoder para a coluna 'label' (tipo de cultura)
print("\n--- Codificando a variável 'label' com LabelEncoder ---")
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Criando um dicionário para mapear os nomes das culturas para seus rótulos numéricos,
# o que facilita a interpretação posterior
labels_dict = dict(zip(le.classes_, le.transform(le.classes_)))
print("Mapeamento dos rótulos:", labels_dict)

# 5. Normalização dos dados numéricos
# Usando StandardScaler para as variáveis numéricas
print("\n--- Normalizando as variáveis numéricas com StandardScaler ---")
# Separando as features (X) e o target (y)
# Excluímos a coluna 'label' original e a 'label_encoded' que é o nosso target
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'humidity_temp_ratio', 'NPK_index']
X = df[features]
y = df['label_encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Criando um novo DataFrame para os dados escalados para facilitar a visualização
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
print(X_scaled_df.head())

# 6. Divisão dos dados em conjuntos de treino e teste (80%/20%)
print("\n--- Dividindo os dados em conjuntos de treino e teste ---")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Tamanho do conjunto de treino (X_train): {X_train.shape}")
print(f"Tamanho do conjunto de teste (X_test): {X_test.shape}")
print(f"Tamanho do conjunto de treino (y_train): {y_train.shape}")
print(f"Tamanho do conjunto de teste (y_test): {y_test.shape}")

print("\n--- Pré-processamento concluído com sucesso! ---")


sns.set(style="whitegrid", palette="muted")

fig, axes = plt.subplots(2, 2, figsize=(14,10))  # 2 linhas, 2 colunas

# 1. Distribuição das culturas
sns.countplot(y=df['label'], order=df['label'].value_counts().index, palette="viridis", ax=axes[0,0])
axes[0,0].set_title("Distribuição das Culturas")

# 2. Heatmap de correlação
sns.heatmap(df[features].corr(), annot=False, cmap="coolwarm", ax=axes[0,1])
axes[0,1].set_title("Matriz de Correlação")

# 3. Boxplot de Nitrogênio
sns.boxplot(data=df, x="label", y="N", ax=axes[1,0])
axes[1,0].set_title("Distribuição de Nitrogênio por Cultura")
axes[1,0].tick_params(axis='x', rotation=90)

# 4. Scatter temperatura x umidade
sns.scatterplot(data=df, x="temperature", y="humidity", hue="label", alpha=0.7, ax=axes[1,1])
axes[1,1].set_title("Temperatura vs Umidade")

plt.tight_layout()
plt.show()
