import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def carregar_stopwords(caminho_arquivo):
    """Lê as stopwords do arquivo .txt externo disponibilizado em aula."""
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())
    return stopwords

def pre_processar_texto_w2v(texto, stopwords):
    """Limpeza focada em preparar tokens para o Word2Vec."""
    texto = str(texto).lower()
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto) # Remove URLs
    texto = re.sub(r'[^\w\s]', '', texto)               # Remove pontuação
    texto = re.sub(r'\d+', '', texto)                   # Remove números
    
    tokens = [palavra for palavra in texto.split() if palavra not in stopwords]
    return tokens

def texto_para_vetor_w2v(tokens, modelo_w2v):
    """Gera um único vetor representativo tirando a média dos vetores das palavras."""
    vetores = [modelo_w2v.wv[palavra] for palavra in tokens if palavra in modelo_w2v.wv]
    if len(vetores) == 0:
        return np.zeros(modelo_w2v.vector_size)
    return np.mean(vetores, axis=0)

def executar_pipeline_avancado():
    # Definição dos caminhos portáveis
    caminho_csv = os.path.join('..', 'dataset', 'dataset_epidemias.csv')
    caminho_stopwords = os.path.join('..', 'dataset', 'stopwords_pt.txt')
    diretorio_resultados = os.path.join('..', 'resultados')
    
    # Garante que a pasta de resultados exista no ambiente
    os.makedirs(diretorio_resultados, exist_ok=True)
    
    print("[1/6] Carregando dados e Stopwords...")
    df = pd.read_csv(caminho_csv)
    stopwords = carregar_stopwords(caminho_stopwords)
    
    print("[2/6] Pré-processando textos...")
    # Une título e texto para extrair o máximo de contexto semântico
    df['conteudo_completo'] = df['titulo'] + " " + df['texto']
    df['tokens'] = df['conteudo_completo'].apply(lambda t: pre_processar_texto_w2v(t, stopwords))
    
    print("[3/6] Treinando modelo Word2Vec baseado no vocabulário local...")
    w2v_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, workers=4)
    
    # Vetorização da base de dados
    X_vetorizado = np.array([texto_para_vetor_w2v(tokens, w2v_model) for tokens in df['tokens']])
    y = df['classificacao'].values
    
    print("[4/6] Dividindo a base e aplicando SMOTE (Apenas no Treino)...")
    # Divisão escalonada (stratify garante proporções iguais de classes no treino e teste)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_vetorizado, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Ajuste dinâmico do parâmetro k_neighbors para não quebrar em bases muito pequenas de teste
    frequencia_classes = pd.Series(y_treino).value_counts()
    menor_classe_treino = min(frequencia_classes)
    k_vizinhos = min(5, menor_classe_treino - 1) if menor_classe_treino > 1 else 1
    
    # O SMOTE entra aqui: expandindo APENAS o ambiente de treino
    smote = SMOTE(random_state=42, k_neighbors=k_vizinhos)
    X_treino_bal, y_treino_bal = smote.fit_resample(X_treino, y_treino)
    
    print("[5/6] Treinando Classificador Naive Bayes Gaussiano...")
    modelo_nb = GaussianNB()
    modelo_nb.fit(X_treino_bal, y_treino_bal)
    
    print("[6/6] Avaliando o modelo e exportando artefatos...")
    y_pred = modelo_nb.predict(X_teste)
    
    acuracia = accuracy_score(y_teste, y_pred)
    matriz = confusion_matrix(y_teste, y_pred)
    relatorio = classification_report(y_teste, y_pred, target_names=['Fake (0)', 'Fato (1)'], zero_division=0)
    
    # Renderização e salvamento da Matriz de Confusão
    plt.figure(figsize=(6, 4))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão: W2V + SMOTE + Naive Bayes')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.tight_layout()
    plt.savefig(os.path.join(diretorio_resultados, 'matriz_w2v_smote.png'), dpi=300)
    plt.close()
    
    # Geração do relatório em formato textual limpo
    with open(os.path.join(diretorio_resultados, 'relatorio_w2v_smote.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Métricas de Avaliação do Modelo\n")
        f.write(f"Acurácia Geral: {acuracia * 100:.2f}%\n\n")
        f.write("Relatório de Classificação Detalhado:\n")
        f.write(relatorio)
        
    print(f"[SUCESSO] Pipeline executado. Resultados salvos em: '{diretorio_resultados}/'")

if __name__ == "__main__":
    # Garante a execução orientada ao diretório do script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    executar_pipeline_avancado()
