import os
import time
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 扩充后的文档列表,覆盖更多K8s核心概念
URLS = [
    "https://kubernetes.io/docs/concepts/overview/what-is-kubernetes/",
    "https://kubernetes.io/docs/concepts/workloads/pods/",
    "https://kubernetes.io/docs/concepts/services-networking/service/",
    "https://kubernetes.io/docs/concepts/storage/persistent-volumes/",
    "https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/replicaset/",
    "https://kubernetes.io/docs/concepts/configuration/configmap/",
    "https://kubernetes.io/docs/concepts/configuration/secret/",
    "https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/job/",
    "https://kubernetes.io/docs/concepts/services-networking/ingress/",
    "https://kubernetes.io/docs/concepts/services-networking/network-policies/",
    "https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/",
    "https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/",
    "https://kubernetes.io/docs/reference/access-authn-authz/rbac/",
    "https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/",
    "https://kubernetes.io/docs/concepts/policy/resource-quotas/",
]

SAVE_DIR = "data/raw"

def make_session() -> requests.Session:
    """创建一个带自动重试能力的请求会话,遇到偶发网络错误会自动重试几次再放弃"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,                # 最多重试3次
        backoff_factor=2,       # 每次重试间隔递增(2秒、4秒、8秒)
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session = make_session()

def fetch_and_clean(url: str) -> str:
    """请求网页,提取正文文字,去掉HTML标签和多余空白"""
    resp = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    main_content = soup.find("main")
    if main_content is None:
        main_content = soup

    text = main_content.get_text(separator="\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    success_count = 0
    failed_urls = []

    for i, url in enumerate(URLS):
        print(f"正在抓取 ({i+1}/{len(URLS)}): {url}")
        try:
            text = fetch_and_clean(url)
            filename = url.rstrip("/").split("/")[-1] or f"doc_{i}"
            filepath = os.path.join(SAVE_DIR, f"{filename}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"已保存: {filepath} ({len(text)} 字符)")
            success_count += 1
        except Exception as e:
            print(f"抓取失败 {url}: {e}")
            failed_urls.append(url)
        time.sleep(1.5)

    print(f"\n抓取完成: 成功 {success_count}/{len(URLS)}")
    if failed_urls:
        print("失败的URL:")
        for u in failed_urls:
            print(f"  - {u}")

if __name__ == "__main__":
    main()