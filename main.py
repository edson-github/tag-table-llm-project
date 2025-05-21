import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def carregar_tabela(caminho_arquivo):
    return pd.read_csv(caminho_arquivo)

def gerar_prompt(pergunta, tabela: pd.DataFrame):
    tabela_str = tabela.to_markdown(index=False)
    return f"""
Você é um assistente que responde perguntas com base na tabela abaixo:

{tabela_str}

Pergunta: {pergunta}
Resposta:"""

def responder(pergunta, tabela):
    prompt = gerar_prompt(pergunta, tabela)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    tabela = carregar_tabela("data/products.csv")
    pergunta = input("Digite sua pergunta sobre a tabela: ")
    resposta = responder(pergunta, tabela)
    print("\nResposta do modelo:")
    print(resposta)
