import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'  # 设置HuggingFace镜像端点（已注释）
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()  # 加载环境配置（从.env文件加载环境变量）

# 配置大语言模型：使用DeepSeek的deepseek-chat模型，API密钥从环境变量获取
Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))

# 配置中文嵌入模型：使用BAAI的中文小型嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 加载本地md文件：从指定路径读取markdown文档
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 创建向量存储索引：将文档转换为向量表示并构建索引
index = VectorStoreIndex.from_documents(docs)

# 创建查询引擎：基于索引构建问答系统
query_engine = index.as_query_engine()

# 打印使用的提示模板：查看系统使用的提示词配置
print(query_engine.get_prompts())

# 执行查询：向查询引擎提问并获取答案
print(query_engine.query("文中举了哪些例子?"))