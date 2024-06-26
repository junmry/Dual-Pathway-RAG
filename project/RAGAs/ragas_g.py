from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
import os
_ = load_dotenv(find_dotenv())
api_key = os.environ["OPENAI_API_KEY"]

os.environ["OPENAI_BASE_URL"] = 'https://key.wenwen-ai.com/v1'
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "18851806367"

graph = Neo4jGraph()

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0, openai_api_key= api_key, model="gpt-4" ),
    graph=graph,
    verbose=True,
)
question = "What AI technologies are used in Cisco's latest cybersecurity defense architecture? "
result = chain.run(question)
print(result)
