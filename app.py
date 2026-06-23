import os
import re
import io
import tempfile
import pandas as pd
import psycopg2
from psycopg2 import pool as pg_pool
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

if not all([GOOGLE_API_KEY, TAVILY_API_KEY, TELEGRAM_TOKEN, DB_NAME, DB_USER, DB_PASSWORD]):
    raise EnvironmentError("Faltam variáveis de ambiente no .env")

# ---------------------------------------------------------------------------
# POOL DE CONEXÕES PostgreSQL  (FIX: evita nova conexão a cada chamada)
# ---------------------------------------------------------------------------
db_pool = pg_pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
)

# ---------------------------------------------------------------------------
# CONTROLE DE CUSTO  (FIX: valores corretos para Gemini 2.5 Flash, jun/2025)
# Gemini 2.5 Flash  — non-thinking (temperatura baixa sem budget):
#   input : $0.15 / 1M tokens
#   output: $0.60 / 1M tokens
# Ajuste se ativar thinking (output sobe para $3.50/1M)
# ---------------------------------------------------------------------------
PRECO_GEMINI_INPUT_POR_TOKEN  = 0.15  / 1_000_000
PRECO_GEMINI_OUTPUT_POR_TOKEN = 0.60  / 1_000_000
PRECO_TAVILY_POR_BUSCA        = 0.008

def calcular_e_exibir_custo(evento: dict, buscas_realizadas: int) -> float:
    tokens_entrada = tokens_saida = 0
    if "agent" in evento:
        msg = evento["agent"]["messages"][-1]
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            tokens_entrada = msg.usage_metadata.get("input_tokens", 0)
            tokens_saida   = msg.usage_metadata.get("output_tokens", 0)
        elif hasattr(msg, "response_metadata"):
            uso = msg.response_metadata.get("usage_metadata", {})
            tokens_entrada = uso.get("prompt_token_count", 0)
            tokens_saida   = uso.get("candidates_token_count", 0)

    custo_total = (
        tokens_entrada    * PRECO_GEMINI_INPUT_POR_TOKEN  +
        tokens_saida      * PRECO_GEMINI_OUTPUT_POR_TOKEN +
        buscas_realizadas * PRECO_TAVILY_POR_BUSCA
    )
    print(
        f"\n  💰 [CUSTO DA INTERAÇÃO]\n"
        f"     Tokens entrada : {tokens_entrada:,}\n"
        f"     Tokens saída   : {tokens_saida:,}\n"
        f"     Buscas Tavily  : {buscas_realizadas}\n"
        f"     TOTAL          : ${custo_total:.6f}"
    )
    return custo_total

# ---------------------------------------------------------------------------
# FILTRO ANTI-INJEÇÃO DE PROMPT
# ---------------------------------------------------------------------------
PADROES_PROIBIDOS = [
    r"ignor(e|a|ar)\s+(as\s+)?(instru[cç][oõ]es|regras|prompt|sistema)",
    r"esque[cç](e|a|er)\s+(tudo|as\s+instru[cç][oõ]es|o\s+que\s+foi\s+dito)",
    r"desconsider(e|a|ar)\s+(as\s+)?(instru[cç][oõ]es|regras)",
    r"a\s+partir\s+de\s+agora\s+(voc[eê]\s+[eé]|fa[cç]a|aja|se\s+comporte)",
    r"seu\s+novo\s+(papel|objetivo|comportamento|prompt|sistema)\s+[eé]",
]
REGEX_COMPILADOS = [re.compile(p, re.IGNORECASE) for p in PADROES_PROIBIDOS]

def verificar_injeccao(mensagem: str) -> bool:
    return any(r.search(mensagem) for r in REGEX_COMPILADOS)

# ---------------------------------------------------------------------------
# FERRAMENTAS
# ---------------------------------------------------------------------------
busca_web = TavilySearch(max_results=2)

MAX_CHARS_ARQUIVO = 15_000   # ~3 750 tokens; suficiente e econômico

def analisar_arquivo(caminho_arquivo: str) -> str:
    """
    Lê PDF, CSV ou XLSX e retorna um resumo/trecho do conteúdo.
    Input: caminho absoluto do arquivo no sistema local.
    """
    try:
        caminho = caminho_arquivo.strip().strip("'\"")

        # PDF
        if caminho.endswith(".pdf"):
            loader = PyPDFLoader(caminho)
            texto  = "\n".join(p.page_content for p in loader.load())
            return f"[PDF] {texto[:MAX_CHARS_ARQUIVO]}"

        # CSV
        if caminho.endswith(".csv"):
            df = pd.read_csv(caminho)
            resumo = (
                f"[CSV] Colunas: {list(df.columns)}\n"
                f"Linhas: {len(df)}\n"
                f"Primeiras linhas:\n{df.head(10).to_string(index=False)}"
            )
            return resumo[:MAX_CHARS_ARQUIVO]

        # XLSX / XLS  (FIX: suporte a planilhas Excel)
        if caminho.endswith((".xlsx", ".xls")):
            xl   = pd.ExcelFile(caminho)
            abas = xl.sheet_names
            partes = [f"[XLSX] Abas encontradas: {abas}"]
            for aba in abas[:3]:          # limita às 3 primeiras abas
                df = xl.parse(aba)
                partes.append(
                    f"\nAba '{aba}' — {len(df)} linhas, colunas: {list(df.columns)}\n"
                    f"{df.head(10).to_string(index=False)}"
                )
            return "\n".join(partes)[:MAX_CHARS_ARQUIVO]

        return "Erro: formato não suportado. Use PDF, CSV ou XLSX."
    except Exception as e:
        return f"Erro na leitura do arquivo: {e}"

leitor_arquivos = Tool(
    name="LeitorDeArquivos",
    func=analisar_arquivo,
    description=(
        "Lê arquivos PDF, CSV e XLSX salvos localmente. "
        "O input DEVE ser o caminho absoluto do arquivo (ex: /tmp/relatorio.pdf)."
    ),
)

def consultar_estoque_sementes(cultura: str) -> str:
    """
    Consulta o catálogo de sementes no PostgreSQL.
    Input: nome da cultura (ex: 'Soja', 'Milho') ou 'todas'.
    """
    conn = db_pool.getconn()
    try:
        cursor = conn.cursor()
        cultura_limpa = cultura.strip().lower()

        if cultura_limpa in ("todas", "tudo", ""):
            cursor.execute(
                "SELECT cultura, variedade, ciclo, resistencia_principal, "
                "tipo_solo_ideal, estoque_toneladas, preco_saca_brl "
                "FROM sementes_catalogo ORDER BY cultura, variedade"
            )
        else:
            cursor.execute(
                "SELECT cultura, variedade, ciclo, resistencia_principal, "
                "tipo_solo_ideal, estoque_toneladas, preco_saca_brl "
                "FROM sementes_catalogo WHERE cultura ILIKE %s "
                "ORDER BY variedade",
                (f"%{cultura_limpa}%",),
            )

        resultados = cursor.fetchall()
        cursor.close()

        if not resultados:
            return f"Nenhuma semente encontrada para: '{cultura}'"

        linhas = [
            f"[POSTGRESQL] Catálogo — '{cultura}':\n"
        ]
        for r in resultados:
            sem_estoque = " ⚠️ SEM ESTOQUE" if r[5] == 0 else ""
            linhas.append(
                f"  • {r[0]} / {r[1]} | Ciclo: {r[2]} | "
                f"Resistência: {r[3]} | Solo: {r[4]} | "
                f"Estoque: {r[5]} ton | R$ {r[6]:.2f}/saca{sem_estoque}"
            )
        return "\n".join(linhas)

    except Exception as e:
        return f"Erro no banco de dados: {e}"
    finally:
        db_pool.putconn(conn)   # sempre devolve a conexão ao pool

ferramenta_banco = Tool(
    name="ConsultarBancoSementes",
    func=consultar_estoque_sementes,
    description=(
        "Consulta o banco de dados PostgreSQL com o catálogo de sementes: "
        "variedades, ciclo, resistências, tipo de solo ideal, estoque e preço. "
        "Input DEVE ser a cultura (ex: 'Soja', 'Milho', 'Sorgo') ou 'todas'."
    ),
)

tools = [busca_web, leitor_arquivos, ferramenta_banco]

# ---------------------------------------------------------------------------
# SYSTEM PROMPT  —  passado UMA VEZ via SystemMessage, não repetido
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = SystemMessage(content="""Você é o Agente AgroEstratégico, assistente especializado \
em agronegócio para produtores rurais brasileiros.

REGRAS OBRIGATÓRIAS:
1. Para sementes, variedades, estoque ou características agronômicas → use ConsultarBancoSementes.
2. Para previsão do tempo, clima atual ou cotações de commodities → use TavilySearch.
3. Quando o produtor enviar um arquivo (PDF, CSV, XLSX) → use LeitorDeArquivos com o caminho informado.
4. Cruze os dados: combine clima e preço de mercado (web) com resistências e ciclo das sementes (banco).
5. NUNCA invente dados agronômicos. Se o estoque for 0, sinalize e sugira alternativas disponíveis.
6. Responda sempre em português brasileiro, de forma clara e prática.
""")

# ---------------------------------------------------------------------------
# AGENTE + MEMÓRIA
# FIX: system prompt injetado via prompt param do create_react_agent,
#      não reenviado como mensagem a cada turno → economia real de tokens.
# ---------------------------------------------------------------------------
memoria       = MemorySaver()
llm           = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY,
)
agent_executor = create_react_agent(
    llm,
    tools,
    checkpointer=memoria,
    prompt=SYSTEM_PROMPT,   # system prompt enviado internamente pelo agente, 1x por sessão
)

# ---------------------------------------------------------------------------
# LOGS DE RACIOCÍNIO
# ---------------------------------------------------------------------------
def log_raciocinio(evento: dict) -> int:
    buscas = 0
    for node_name, node_data in evento.items():
        if node_name == "agent":
            msg = node_data["messages"][-1]
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n  💭 [PENSAMENTO] Agente decidiu agir.")
                    print(f"  ⚡ [AÇÃO] Ferramenta: '{tc['name']}' | Input: {tc['args']}")
                    if "tavily" in tc["name"].lower():
                        buscas += 1
            elif msg.content:
                txt = (
                    msg.content if isinstance(msg.content, str)
                    else " ".join(b.get("text", "") for b in msg.content if isinstance(b, dict))
                )
                print(f"\n  [RESPOSTA AGENTE] {txt[:200]}...")
        elif node_name == "tools":
            for msg in node_data["messages"]:
                print(f"\n  🔍 [OBSERVAÇÃO] {str(msg.content)[:300]}...")
    return buscas

def extrair_resposta(node_data: dict) -> str:
    msg = node_data["messages"][-1]
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        return " ".join(b.get("text", "") for b in msg.content if isinstance(b, dict) and "text" in b)
    return ""

# ---------------------------------------------------------------------------
# DOWNLOAD DE ARQUIVO DO TELEGRAM
# FIX: handler completo para Document (PDF, CSV, XLSX)
# ---------------------------------------------------------------------------
EXTENSOES_SUPORTADAS = (".pdf", ".csv", ".xlsx", ".xls")

async def baixar_arquivo_telegram(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str | None:
    """
    Baixa o documento enviado pelo produtor e salva em arquivo temporário.
    Retorna o caminho do arquivo ou None se o formato não for suportado.
    """
    doc = update.message.document
    nome = doc.file_name or ""

    if not nome.lower().endswith(EXTENSOES_SUPORTADAS):
        await update.message.reply_text(
            "⚠️ Formato não suportado. Envie um arquivo PDF, CSV ou XLSX."
        )
        return None

    telegram_file = await context.bot.get_file(doc.file_id)
    sufixo        = os.path.splitext(nome)[1]
    tmp           = tempfile.NamedTemporaryFile(delete=False, suffix=sufixo)
    await telegram_file.download_to_drive(tmp.name)
    tmp.close()
    return tmp.name

# ---------------------------------------------------------------------------
# EXECUÇÃO DO AGENTE
# FIX: thread_id único por usuário (user_id), não por chat_id,
#      para evitar mistura de contexto em grupos.
# ---------------------------------------------------------------------------
async def executar_agente(
    update: Update,
    mensagem_usuario: str,
    user_id: str,
    caminho_arquivo: str | None = None,
):
    try:
        if verificar_injeccao(mensagem_usuario):
            await update.message.reply_text("🚨 Mensagem bloqueada por segurança.")
            return

        await update.message.reply_chat_action(action="typing")

        # Se há arquivo, acrescenta instrução para o agente usar o LeitorDeArquivos
        if caminho_arquivo:
            mensagem_usuario = (
                f"{mensagem_usuario}\n\n"
                f"[ARQUIVO ENVIADO] Use a ferramenta LeitorDeArquivos com o caminho: {caminho_arquivo}"
            )

        # FIX: thread por user_id → memória separada para cada produtor
        config = {"configurable": {"thread_id": f"user_{user_id}"}}

        resposta_final = ""
        total_buscas   = 0
        ultimo_evento  = {}

        for evento in agent_executor.stream(
            {"messages": [("user", mensagem_usuario)]},
            config=config,
            stream_mode="updates",
        ):
            total_buscas += log_raciocinio(evento)
            if "agent" in evento:
                txt = extrair_resposta(evento["agent"])
                if txt:
                    resposta_final = txt
                    ultimo_evento  = evento

        calcular_e_exibir_custo(ultimo_evento, total_buscas)

        if not resposta_final.strip():
            resposta_final = "Não consegui formular uma resposta. Tente reformular a pergunta."

        # Telegram limita mensagens a 4 096 caracteres
        for i in range(0, len(resposta_final), 4096):
            await update.message.reply_text(resposta_final[i:i+4096])

    except Exception as e:
        print(f"[ERRO] {type(e).__name__}: {e}")
        await update.message.reply_text(f"❌ Erro interno: {type(e).__name__}. Tente novamente.")
    finally:
        # Remove arquivo temporário após uso
        if caminho_arquivo and os.path.exists(caminho_arquivo):
            os.remove(caminho_arquivo)

# ---------------------------------------------------------------------------
# HANDLERS DO TELEGRAM
# ---------------------------------------------------------------------------
async def processar_mensagem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para mensagens de texto."""
    await executar_agente(
        update,
        update.message.text,
        str(update.message.from_user.id),   # FIX: user_id, não chat_id
    )

async def processar_documento(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler para arquivos enviados pelo produtor (PDF, CSV, XLSX).
    FIX: fluxo completo de download → leitura → análise pelo agente.
    """
    caminho = await baixar_arquivo_telegram(update, context)
    if not caminho:
        return

    # Legenda da mensagem (se houver) vira a pergunta; senão, usa padrão
    legenda = (update.message.caption or "").strip()
    if not legenda:
        legenda = "Analise o arquivo que enviei e me dê recomendações agronômicas."

    await executar_agente(
        update,
        legenda,
        str(update.message.from_user.id),
        caminho_arquivo=caminho,
    )

async def comando_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🌱 *Agente AgroEstratégico* iniciado!\n\n"
        "Pergunte o que plantar, qual variedade escolher, como está o clima ou "
        "o preço das commodities.\n\n"
        "Você também pode enviar um arquivo *PDF, CSV ou XLSX* para análise.",
        parse_mode="Markdown",
    )

# ---------------------------------------------------------------------------
# INICIALIZAÇÃO
# ---------------------------------------------------------------------------
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", comando_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, processar_mensagem))
    app.add_handler(MessageHandler(filters.Document.ALL, processar_documento))   # FIX: handler de arquivos
    print("🚜 Agente AgroEstratégico iniciado! Conectado ao PostgreSQL via pool.")
    app.run_polling()

if __name__ == "__main__":
    main()