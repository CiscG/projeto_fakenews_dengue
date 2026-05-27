import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def carregar_stopwords(caminho_arquivo):
    """Lê as stopwords tratando as variações de encoding de forma segura."""
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            linhas = f.read().splitlines()
    except UnicodeDecodeError:
        with open(caminho_arquivo, 'r', encoding='iso-8859-1') as f:
            linhas = f.read().splitlines()
    return set(palavra.strip() for palavra in linhas if palavra.strip())

def limpar_texto_para_tfidf(texto, stopwords):
    """Realiza a limpeza do texto mantendo-o em formato de string para o Vectorizer."""
    texto = str(texto).lower()
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto) # Remove URLs
    texto = re.sub(r'[^\w\s]', '', texto)               # Remove pontuação
    texto = re.sub(r'\d+', '', texto)                   # Remove números
    
    # Filtra as stopwords
    palavras = [palavra for palavra in texto.split() if palavra not in stopwords]
    return " ".join(palavras)

def executar_pipeline_tfidf():
    # Configuração dos caminhos portáveis
    caminho_csv = os.path.join('..', 'dataset', 'dataset_epidemias') + '.csv'
    caminho_stopwords = os.path.join('..', 'dataset', 'stopwords_pt.txt')
    diretorio_resultados = os.path.join('..', 'resultados')
    
    os.makedirs(diretorio_resultados, exist_ok=True)
    
    print("[1/5] Carregando dados e tratando cabeçalhos corrompidos...")
    with open(caminho_csv, 'r', encoding='utf-8', errors='ignore') as f:
        primeira_linha = f.readline()
        
    if 'titulo' not in primeira_linha:
        df = pd.read_csv(caminho_csv, skiprows=1)
    else:
        df = pd.read_csv(caminho_csv)
        
    stopwords = carregar_stopwords(caminho_stopwords)
    
    print("[2/5] Pré-processando textos e aplicando limpeza...")
    df['conteudo_completo'] = df['titulo'].fillna('') + " " + df['texto'].fillna('')
    df['texto_limpo'] = df['conteudo_completo'].apply(lambda t: limpar_texto_para_tfidf(t, stopwords))
    
    print("[3/5] Vetorizando com TF-IDF estatístico...")
    # Usando ngram_range=(1,2) para capturar termos compostos como "ministério saúde" ou "pó café"
    vetorizador = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X_tfidf = vetorizador.fit_transform(df['texto_limpo']).toarray()
    y = df['classificacao'].values
    
    print("[4/5] Dividindo a base e aplicando SMOTE adaptativo no treino...")
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_tfidf, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Ajusta o número de vizinhos do SMOTE dinamicamente baseado na menor classe de treino
    frequencia_classes = pd.Series(y_treino).value_counts()
    menor_classe_treino = min(frequencia_classes)
    k_vizinhos = min(5, menor_classe_treino - 1) if menor_classe_treino > 1 else 1
    
    smote = SMOTE(random_state=42, k_neighbors=k_vizinhos)
    X_treino_bal, y_treino_bal = smote.fit_resample(X_treino, y_treino)
    
    print("[5/5] Treinando classificador Naive Bayes Multinomial...")
    # MultinomialNB é o algoritmo padrão e matematicamente correto para matrizes de frequências TF-IDF
    modelo_nb = MultinomialNB()
    modelo_nb.fit(X_treino_bal, y_treino_bal)
    
    # Avaliação do Modelo
    y_pred = modelo_nb.predict(X_teste)
    acuracia = accuracy_score(y_teste, y_pred)
    matriz = confusion_matrix(y_teste, y_pred)
    relatorio = classification_report(y_teste, y_pred, target_names=['Fake (0)', 'Fato (1)'], zero_division=0)
    
    # Exportação gráfica da Matriz de Confusão
    plt.figure(figsize=(6, 4))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Greens')
    plt.title('Matriz de Confusão: TF-IDF + Naive Bayes')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.tight_layout()
    plt.savefig(os.path.join(diretorio_resultados, 'matriz_tfidf_nb.png'), dpi=300)
    plt.close()
    
    # Gravação das métricas em relatório textual
    caminho_relatorio = os.path.join(diretorio_resultados, 'relatorio_tfidf_nb.txt')
    with open(caminho_relatorio, 'w', encoding='utf-8') as f:
        f.write("Métricas de Avaliação do Modelo (Abordagem Estatística TF-IDF)\n")
        f.write(f"Acurácia Geral: {acuracia * 100:.2f}%\n\n")
        f.write("Relatório de Classificação Detalhado:\n")
        f.write(relatorio)
        
    print(f"\n[SUCESSO] Pipeline concluído! Resultados salvos na pasta '{diretorio_resultados}/'")
    print(f"Confira o arquivo: 'relatorio_tfidf_nb.txt'")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    executar_pipeline_tfidf()