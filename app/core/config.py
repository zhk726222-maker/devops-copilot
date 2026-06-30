import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI

# 加载 .env 文件里的环境变量
load_dotenv()

# 统一管理的配置项
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
DEFAULT_MODEL = "glm-4.6"
EMBEDDING_MODEL = "embedding-3"

# 全局共用的智谱客户端,其他文件直接从这里导入,不用重复初始化
client = ZhipuAI(api_key=ZHIPU_API_KEY)