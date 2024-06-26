from dotenv import find_dotenv, load_dotenv
import os
from llm.call_llm import get_completion

from langchain_community.graphs import Neo4jGraph

# Warning control
import warnings
warnings.filterwarnings("ignore")


_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]
#get_completion(api_key=openai_api_key, prompt="你是谁", model="gpt-3.5-turbo" )

kg = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="18851806367")


