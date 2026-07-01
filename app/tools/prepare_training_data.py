import json
import os

# 手工标注的训练样本,覆盖三类路由场景,尽量多样化措辞和表达方式
TRAINING_SAMPLES = [
    # RAG类 - K8s概念/原理/用法
    {"query": "什么是Kubernetes", "label": "RAG"},
    {"query": "Pod和容器有什么区别", "label": "RAG"},
    {"query": "Deployment和ReplicaSet是什么关系", "label": "RAG"},
    {"query": "ConfigMap怎么用", "label": "RAG"},
    {"query": "什么是Service的ClusterIP类型", "label": "RAG"},
    {"query": "污点和容忍度是干什么用的", "label": "RAG"},
    {"query": "StatefulSet和Deployment有什么区别", "label": "RAG"},
    {"query": "Ingress是什么", "label": "RAG"},
    {"query": "HPA是怎么工作的", "label": "RAG"},
    {"query": "RBAC的ClusterRole和Role有什么区别", "label": "RAG"},
    {"query": "liveness probe和readiness probe的区别", "label": "RAG"},
    {"query": "NetworkPolicy怎么限制Pod之间的通信", "label": "RAG"},
    {"query": "PersistentVolume的回收策略有哪几种", "label": "RAG"},
    {"query": "kube-scheduler是怎么选节点的", "label": "RAG"},
    {"query": "DaemonSet和Deployment有什么区别", "label": "RAG"},
    {"query": "Kubernetes命名空间的作用是什么", "label": "RAG"},
    {"query": "Job和CronJob的区别", "label": "RAG"},
    {"query": "什么是资源配额", "label": "RAG"},
    {"query": "Secret和ConfigMap的区别", "label": "RAG"},
    {"query": "如何实现滚动更新", "label": "RAG"},
    {"query": "Pod的生命周期是怎样的", "label": "RAG"},
    {"query": "什么是sidecar容器", "label": "RAG"},
    {"query": "k8s的调度器工作原理", "label": "RAG"},
    {"query": "怎么理解容器化和虚拟机的区别", "label": "RAG"},
    {"query": "ReadWriteOnce和ReadWriteMany有什么区别", "label": "RAG"},

    # SQL类 - 查询运维数据
    {"query": "现在有哪些服务在告警", "label": "SQL"},
    {"query": "谁在负责值班", "label": "SQL"},
    {"query": "上周有几次部署失败了", "label": "SQL"},
    {"query": "payment-api最近一次是谁部署的", "label": "SQL"},
    {"query": "critical级别的告警有多少条", "label": "SQL"},
    {"query": "production环境有几个服务", "label": "SQL"},
    {"query": "今天有哪些服务发生过告警", "label": "SQL"},
    {"query": "最近30天部署最频繁的服务是哪个", "label": "SQL"},
    {"query": "状态异常的服务有哪些", "label": "SQL"},
    {"query": "下周值班的是谁", "label": "SQL"},
    {"query": "user-auth部署了几个版本", "label": "SQL"},
    {"query": "有哪些告警还没有解决", "label": "SQL"},
    {"query": "staging环境有哪些服务", "label": "SQL"},
    {"query": "最近一次部署是什么时候", "label": "SQL"},
    {"query": "哪个工程师部署次数最多", "label": "SQL"},
    {"query": "billing-service有没有出现过critical告警", "label": "SQL"},
    {"query": "当前值班的团队是哪个", "label": "SQL"},
    {"query": "过去7天告警最多的服务是哪个", "label": "SQL"},
    {"query": "有多少服务是StatefulSet类型的", "label": "SQL"},
    {"query": "notification-worker部署失败过几次", "label": "SQL"},
    {"query": "现在有几个服务是degraded状态", "label": "SQL"},
    {"query": "本月有哪些回滚操作", "label": "SQL"},
    {"query": "search-engine最近的告警内容是什么", "label": "SQL"},
    {"query": "哪些服务的副本数少于3个", "label": "SQL"},
    {"query": "SRE团队下周什么时候值班", "label": "SQL"},

    # TOOL类 - 执行操作/动作
    {"query": "帮我重启一下payment-api", "label": "TOOL"},
    {"query": "查一下user-auth的日志", "label": "TOOL"},
    {"query": "把billing-service扩容到5个副本", "label": "TOOL"},
    {"query": "看一下notification-worker的CPU占用", "label": "TOOL"},
    {"query": "重启order-service", "label": "TOOL"},
    {"query": "查看inventory-api最近20行日志", "label": "TOOL"},
    {"query": "把search-engine缩容到2个副本", "label": "TOOL"},
    {"query": "帮我看看recommendation-engine的内存使用情况", "label": "TOOL"},
    {"query": "重新拉起image-processor服务", "label": "TOOL"},
    {"query": "查一下log-aggregator的运行状态", "label": "TOOL"},
    {"query": "把payment-api的副本数调整到3", "label": "TOOL"},
    {"query": "看看user-auth现在跑了几个Pod", "label": "TOOL"},
    {"query": "帮我把order-service重启一下", "label": "TOOL"},
    {"query": "查看billing-service最近的错误日志", "label": "TOOL"},
    {"query": "把inventory-api扩到4个副本", "label": "TOOL"},
    {"query": "看一下payment-api的资源占用", "label": "TOOL"},
    {"query": "重启所有degraded状态的服务", "label": "TOOL"},
    {"query": "查看notification-worker的内存用量", "label": "TOOL"},
    {"query": "帮我把search-engine副本数改成3", "label": "TOOL"},
    {"query": "拉取一下image-processor最新的日志", "label": "TOOL"},
    {"query": "检查一下log-aggregator的CPU使用率", "label": "TOOL"},
    {"query": "把recommendation-engine重启", "label": "TOOL"},
    {"query": "看看billing-service现在有几个副本在跑", "label": "TOOL"},
    {"query": "帮我扩容user-auth到6个实例", "label": "TOOL"},
    {"query": "查一下order-service的健康状态", "label": "TOOL"},
]

def prepare_data():
    """把训练样本转换成微调需要的对话格式(ChatML格式),保存成jsonl文件"""
    os.makedirs("data/training", exist_ok=True)
    output_path = "data/training/routing_train.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in TRAINING_SAMPLES:
            # 构造成"系统指令+用户问题+模型回答"的对话格式
            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个任务路由助手。判断用户的问题应该交给哪个子系统处理:RAG(知识问答)、SQL(数据查询)、TOOL(执行操作)。只回答一个词。"
                    },
                    {
                        "role": "user",
                        "content": sample["query"]
                    },
                    {
                        "role": "assistant",
                        "content": sample["label"]
                    }
                ]
            }
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    print(f"训练数据已生成: {output_path}")
    print(f"总样本数: {len(TRAINING_SAMPLES)}")

    # 统计各类别分布
    from collections import Counter
    label_counts = Counter(s["label"] for s in TRAINING_SAMPLES)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}条")

if __name__ == "__main__":
    prepare_data()