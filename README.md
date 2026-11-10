# Sistema de Recuperação de Informações com FAISS e RetrievalQA

Atividade da disciplina **Tópicos Específicos em Sistemas para Internet I** (TSI/IFRN).

## Descrição do projeto

Este projeto é um sistema simples de perguntas e respostas sobre Inteligência Artificial. Ele usa um banco de dados vetorial (FAISS) para guardar os embeddings de um texto e o módulo **RetrievalQA** do LangChain para responder as perguntas com base nesse texto.

O texto usado está em `data/textos_ia.txt` e fala sobre IA, aprendizado de máquina, redes neurais, transformers, embeddings, bancos vetoriais e RAG.

Tudo roda localmente com o **Ollama**, então não precisa de chave de API:

- `mxbai-embed-large` para gerar os embeddings (vetores de 1024 posições)
- `llama3.2` para escrever a resposta

### Como funciona

1. O script lê o arquivo de texto.
2. O texto é dividido em chunks com o `RecursiveCharacterTextSplitter` (chunk de 800 caracteres e overlap de 50).
3. Cada chunk vira um embedding com o modelo `mxbai-embed-large`.
4. Os embeddings são inseridos no FAISS, e o índice é salvo na pasta `faiss_index/` para não precisar gerar tudo de novo a cada execução.
5. Quando o usuário faz uma pergunta:
   - a pergunta é convertida em embedding;
   - o FAISS busca os 3 chunks mais próximos;
   - o RetrievalQA manda esses chunks junto com a pergunta para o Llama 3.2, que monta a resposta.

O script mostra cada etapa no terminal (quantidade de chunks, tamanho do embedding, vetores no FAISS, distância dos chunks encontrados), assim dá para ver o que está acontecendo por trás.

## Estrutura

```
├── data/
│   └── textos_ia.txt     # texto usado como base de conhecimento
├── sistema_rag.py        # código do sistema
├── requirements.txt      # bibliotecas necessárias
└── README.md
```

## Como instalar e executar

### 1. Instalar o Ollama e baixar os modelos

Baixe o Ollama em https://ollama.com/download e instale. Depois, no terminal:

```
ollama pull mxbai-embed-large
ollama pull llama3.2
```

Para conferir se os dois modelos foram baixados:

```
ollama list
```

### 2. Clonar o repositório

```
git clone https://github.com/SEU-USUARIO/atividade-faiss-retrievalqa.git
cd atividade-faiss-retrievalqa
```

### 3. Criar o ambiente virtual e instalar as dependências

No Windows (PowerShell):

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

No Linux ou Mac:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Obs.: o `requirements.txt` fixa o LangChain na versão 0.3, porque nas versões 1.x o `RetrievalQA` saiu do pacote principal.

### 4. Executar

Modo interativo (você digita as perguntas e escreve `sair` para terminar):

```
python sistema_rag.py
```

Fazer uma pergunta só:

```
python sistema_rag.py --pergunta "O que é o FAISS?"
```

Recriar o banco vetorial do zero (por exemplo, depois de mudar o texto ou o tamanho do chunk):

```
python sistema_rag.py --recriar
```

Na primeira execução o banco vetorial é criado automaticamente. Nas próximas ele é carregado da pasta `faiss_index/`.

Para usar outro texto, é só trocar o arquivo em `data/` (ou mudar a variável `ARQUIVO_TEXTO` no código) e rodar com `--recriar`.

## Exemplo de execução

```
[1] Texto carregado: data/textos_ia.txt (7425 caracteres)
[2] Quantidade de chunks gerados: 12
[3] Embedding do chunk 1 tem 1024 dimensões
[4] Gerando embeddings de todos os chunks e inserindo no FAISS...
    Vetores inseridos no FAISS: 12
    Banco vetorial salvo na pasta 'faiss_index/'

======================================================================
Pergunta: O que é o FAISS?
[a] Pergunta convertida em embedding (1024 dimensões)
[b] 3 chunks mais próximos encontrados no FAISS:
    1. distância=... | Bancos de Dados Vetoriais Um banco de dados vetorial guarda embeddings...
    ...
[c] Gerando resposta com o RetrievalQA...

Resposta: O FAISS (Facebook AI Similarity Search) é uma biblioteca criada pelo Facebook, hoje Meta, em 2017, para fazer buscas de similaridade entre vetores de forma rápida...
```

## Exemplos de perguntas e respostas

Respostas reais geradas pelo sistema (Llama 3.2 com temperatura 0). Entre parênteses está o chunk mais próximo encontrado no FAISS e a distância dele (quanto menor, mais parecido com a pergunta).

**Pergunta:** O que é o FAISS?
**Chunk mais próximo:** Bancos de Dados Vetoriais (distância 0.7911)
**Resposta:** O FAISS é uma biblioteca criada pelo Facebook, hoje Meta, em 2017, para fazer buscas de similaridade entre vetores de forma muito rápida, inclusive usando GPU, e consegue trabalhar com bilhões de vetores.

**Pergunta:** Quem criou o Perceptron e em que ano?
**Chunk mais próximo:** Redes Neurais Artificiais (distância 0.5239)
**Resposta:** O Perceptron foi criado por Frank Rosenblatt em 1958.

**Pergunta:** Em que artigo a arquitetura Transformer foi apresentada?
**Chunk mais próximo:** Transformers (distância 0.2705)
**Resposta:** A arquitetura Transformer foi apresentada no artigo "Attention Is All You Need" (Atenção é tudo o que você precisa), de Vaswani e outros pesquisadores do Google, em 2017.

**Pergunta:** O que acontece se o chunk for muito pequeno?
**Chunk mais próximo:** Chunking (distância 0.4872)
**Resposta:** Se o chunk for muito pequeno, pode cortar a informação no meio e fazer o modelo responder que não sabe.

**Pergunta:** Qual é a capital da França?
**Chunk mais próximo:** Transformers (distância 0.9586)
**Resposta:** Não sabe.

Essa última pergunta foi feita de propósito, porque o assunto não está no texto. Todos os chunks encontrados ficaram com distância perto de 1, bem longe da pergunta, e o modelo respondeu que não sabe em vez de inventar. Isso acontece porque o prompt manda usar somente as informações do contexto recuperado do banco vetorial.

## Tecnologias

- Python 3
- LangChain (`RetrievalQA`, `RecursiveCharacterTextSplitter`, `TextLoader`)
- FAISS (`faiss-cpu`)
- Ollama (`mxbai-embed-large` e `llama3.2`)

## Referências

- Código base da disciplina: https://github.com/tmedeirosb/tsi-llm/blob/main/vb-faiss-un04.ipynb
- Documentação do LangChain: https://python.langchain.com
- FAISS: https://github.com/facebookresearch/faiss
- Ollama: https://ollama.com
