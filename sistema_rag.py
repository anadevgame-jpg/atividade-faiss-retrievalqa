"""
Sistema de Recuperação de Informações com FAISS e RetrievalQA
Disciplina: Tópicos Específicos em Sistemas para Internet I (TSI/IFRN)

Fluxo do sistema:
  1. Carrega o texto da pasta data/
  2. Divide o texto em chunks
  3. Gera um embedding para cada chunk (Ollama + mxbai-embed-large)
  4. Guarda os embeddings no banco vetorial FAISS
  5. Recebe a pergunta do usuário, converte em embedding, busca os chunks
     mais próximos e o RetrievalQA (com o Llama 3.2) monta a resposta

Uso:
  python sistema_rag.py                 -> modo interativo (digita as perguntas)
  python sistema_rag.py --recriar       -> apaga e cria o banco vetorial de novo
  python sistema_rag.py --pergunta "O que é FAISS?"
"""

import argparse
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Configurações (dá para mudar aqui e testar outros valores)
# ---------------------------------------------------------------------------
ARQUIVO_TEXTO = Path("data/textos_ia.txt")
PASTA_INDICE = Path("faiss_index")      # onde o banco vetorial fica salvo

MODELO_EMBEDDING = "mxbai-embed-large"  # gera vetores de 1024 posições
MODELO_LLM = "llama3.2"                 # modelo que escreve a resposta

CHUNK_SIZE = 800     # tamanho máximo de cada chunk (em caracteres)
CHUNK_OVERLAP = 50   # quantos caracteres um chunk repete do anterior
K_DOCUMENTOS = 3     # quantos chunks o retriever devolve para o LLM


# ---------------------------------------------------------------------------
# Etapa 1 e 2: carregar o texto e dividir em chunks
# ---------------------------------------------------------------------------
def carregar_e_dividir(caminho: Path):
    """Lê o arquivo de texto e divide em chunks."""
    loader = TextLoader(str(caminho), encoding="utf-8")
    documentos = loader.load()

    divisor = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = divisor.split_documents(documentos)

    print(f"[1] Texto carregado: {caminho} ({len(documentos[0].page_content)} caracteres)")
    print(f"[2] Quantidade de chunks gerados: {len(chunks)}")
    print(f"    Exemplo (chunk 1): {chunks[0].page_content[:150]}...\n")
    return chunks


# ---------------------------------------------------------------------------
# Etapa 3 e 4: gerar embeddings e inserir no FAISS
# ---------------------------------------------------------------------------
def criar_banco_vetorial(chunks, embeddings):
    """Gera os embeddings de cada chunk e guarda no FAISS."""
    # Mostra um embedding de exemplo para ver como ele é
    vetor_exemplo = embeddings.embed_query(chunks[0].page_content)
    print(f"[3] Embedding do chunk 1 tem {len(vetor_exemplo)} dimensões")
    print(f"    Primeiros valores: {[round(float(v), 4) for v in vetor_exemplo[:5]]}\n")

    # from_documents gera o embedding de todos os chunks e cria o índice FAISS
    print("[4] Gerando embeddings de todos os chunks e inserindo no FAISS...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(PASTA_INDICE))
    print(f"    Vetores inseridos no FAISS: {db.index.ntotal}")
    print(f"    Banco vetorial salvo na pasta '{PASTA_INDICE}/'\n")
    return db


def carregar_ou_criar_banco(embeddings, recriar=False):
    """Usa o banco salvo se ele já existir; se não, cria um novo."""
    # Confere se o arquivo do índice existe (e não só a pasta).
    # Com --recriar, o banco é gerado de novo e os arquivos são sobrescritos.
    indice_existe = (PASTA_INDICE / "index.faiss").exists()

    if indice_existe and not recriar:
        db = FAISS.load_local(
            str(PASTA_INDICE),
            embeddings,
            allow_dangerous_deserialization=True,  # o índice foi criado por nós mesmos
        )
        print(f"Banco vetorial carregado de '{PASTA_INDICE}/' ({db.index.ntotal} vetores)")
        print("(use --recriar para gerar os embeddings de novo)\n")
        return db

    chunks = carregar_e_dividir(ARQUIVO_TEXTO)
    return criar_banco_vetorial(chunks, embeddings)


# ---------------------------------------------------------------------------
# Etapa 5: RetrievalQA
# ---------------------------------------------------------------------------
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Você é um assistente que responde perguntas sobre Inteligência Artificial.\n"
        "Use somente as informações do contexto abaixo para responder, em português.\n"
        "Se a resposta não estiver no contexto, diga apenas que não sabe.\n\n"
        "Contexto:\n{context}\n\n"
        "Pergunta: {question}\n"
        "Resposta:"
    ),
)


def criar_qa_chain(db):
    """Liga o RetrievalQA ao banco vetorial FAISS."""
    # temperature=0: o modelo pega sempre as palavras mais prováveis, então a
    # resposta fica mais conservadora e fiel ao texto (temperaturas altas,
    # perto de 2, deixam o texto "criativo" demais e até incompreensível)
    llm = OllamaLLM(model=MODELO_LLM, temperature=0)
    retriever = db.as_retriever(search_kwargs={"k": K_DOCUMENTOS})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",              # coloca os chunks encontrados direto no prompt
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,    # devolve também os chunks usados
    )
    return qa_chain


def responder(pergunta, db, embeddings, qa_chain):
    """Mostra cada passo da busca e a resposta final."""
    print("=" * 70)
    print(f"Pergunta: {pergunta}")

    # Passo a: a pergunta vira um embedding
    vetor_pergunta = embeddings.embed_query(pergunta)
    print(f"[a] Pergunta convertida em embedding ({len(vetor_pergunta)} dimensões)")

    # Passo b: busca os chunks mais próximos no FAISS
    # (o score é a distância: quanto MENOR, mais parecido)
    resultados = db.similarity_search_with_score_by_vector(vetor_pergunta, k=K_DOCUMENTOS)
    print(f"[b] {K_DOCUMENTOS} chunks mais próximos encontrados no FAISS:")
    for i, (doc, distancia) in enumerate(resultados, start=1):
        trecho = doc.page_content[:90].replace("\n", " ")
        print(f"    {i}. distância={distancia:.4f} | {trecho}...")

    # Passo c: o RetrievalQA envia os chunks + pergunta para o LLM
    print("[c] Gerando resposta com o RetrievalQA...\n")
    saida = qa_chain.invoke({"query": pergunta})
    print(f"Resposta: {saida['result'].strip()}\n")
    return saida


def main():
    parser = argparse.ArgumentParser(description="Sistema RAG com FAISS e RetrievalQA")
    parser.add_argument("--recriar", action="store_true", help="recria o banco vetorial")
    parser.add_argument("--pergunta", type=str, help="faz uma única pergunta e encerra")
    args = parser.parse_args()

    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
    db = carregar_ou_criar_banco(embeddings, recriar=args.recriar)
    qa_chain = criar_qa_chain(db)

    if args.pergunta:
        responder(args.pergunta, db, embeddings, qa_chain)
        return

    print("Digite sua pergunta (ou 'sair' para encerrar).")
    while True:
        pergunta = input("\n> ").strip()
        if pergunta.lower() in ("sair", "exit", "q"):
            print("Até mais!")
            break
        if pergunta:
            responder(pergunta, db, embeddings, qa_chain)


if __name__ == "__main__":
    try:
        main()
    except ConnectionError:
        print("\nERRO: não consegui falar com o Ollama. Abra o Ollama ou rode 'ollama serve'.")
    except Exception as erro:
        if "not found" in str(erro).lower():
            print(f"\nERRO: modelo não encontrado. Rode 'ollama pull {MODELO_EMBEDDING}' "
                  f"e 'ollama pull {MODELO_LLM}'.\nDetalhe: {erro}")
        else:
            raise
