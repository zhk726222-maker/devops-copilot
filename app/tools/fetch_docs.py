import os
import time
import requests
from bs4 import BeautifulSoup

# 先用几篇K8s核心概念文档练手,后面可以扩充更多
URLS = [
    "https://kubernetes.io/docs/concepts/overview/what-is-kubernetes/",
    "https://kubernetes.io/docs/concepts/workloads/pods/",
    "https://kubernetes.io/docs/concepts/services-networking/service/",
    "https://kubernetes.io/docs/concepts/storage/persistent-volumes/",
    "https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/",
]

SAVE_DIR = "data/raw"

def fetch_and_clean(url: str) -> str:
    """请求网页,提取正文文字,去掉HTML标签和多余空白"""
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # K8s文档正文一般在 <main> 标签里,去掉导航栏、侧边栏这些干扰内容
    main_content = soup.find("main")
    if main_content is None:
        main_content = soup

    text = main_content.get_text(separator="\n")
    # 把多个连续空行压缩成一个
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    for i, url in enumerate(URLS):
        print(f"正在抓取: {url}")
        try:
            text = fetch_and_clean(url)
            filename = url.rstrip("/").split("/")[-1] or f"doc_{i}"
            filepath = os.path.join(SAVE_DIR, f"{filename}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"已保存: {filepath} ({len(text)} 字符)")
        except Exception as e:
            print(f"抓取失败 {url}: {e}")
        time.sleep(1)  # 礼貌性延迟,避免请求太快被对方网站限流

if __name__ == "__main__":
    main()

main()