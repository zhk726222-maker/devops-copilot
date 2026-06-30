"""
一键重建知识库脚本
用法: python -m app.tools.build_knowledge_base

执行顺序: 抓取文档 -> 切块(含异常文本清洗) -> 向量化入库
"""
import sys
from app.tools.fetch_docs import main as fetch_docs
from app.tools.vectorstore import build_index

def build_knowledge_base(skip_fetch: bool = False):
    """
    重建整个知识库
    skip_fetch: 如果 data/raw 下已经有文档了,设为True可以跳过重新抓取,
                只重新切块和入库(比如改了清洗规则,不想重新爬网页时用)
    """
    if not skip_fetch:
        print("===== 第1步: 抓取文档 =====")
        fetch_docs()
    else:
        print("===== 跳过抓取,使用已有文档 =====")

    print("\n===== 第2步+第3步: 切块并向量化入库 =====")
    build_index()

    print("\n===== 知识库重建完成 =====")

if __name__ == "__main__":
    skip = "--skip-fetch" in sys.argv
    build_knowledge_base(skip_fetch=skip)