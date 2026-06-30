from app.tools.hybrid_search import hybrid_search

query = "What is the right way to store sensitive data like passwords for my application"
candidates = hybrid_search(query, top_k=10, candidate_k=10)

print(f"找到 {len(candidates)} 个候选\n")

for i, c in enumerate(candidates, start=1):
    text = c["text"]
    length = len(text)
    # 检查是否有异常长的连续无空格字符串(比如base64编码、长hash值)
    longest_token = max((len(w) for w in text.split()), default=0)
    print(f"候选{i}: id={c['id']}, 来源={c['source']}, 文本长度={length}, 最长单词长度={longest_token}")
    if longest_token > 50:
        print(f"   ⚠️ 检测到异常长片段: {[w for w in text.split() if len(w) > 50][:3]}")