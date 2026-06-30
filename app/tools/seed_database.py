import random
import datetime
from app.core.database import SessionLocal, Service, Alert, Deployment, OnCall, init_db

# 固定随机种子,保证每次生成的数据是一样的(方便复现和调试,不会每次跑结果都不同)
random.seed(42)

SERVICE_NAMES = [
    "payment-api", "user-auth", "order-service", "inventory-api",
    "notification-worker", "search-engine", "recommendation-engine",
    "image-processor", "log-aggregator", "billing-service",
]
NAMESPACES = ["production", "staging", "production", "production"]  # 故意让production出现概率更高,更贴近真实分布
ENGINEERS = ["张伟", "李娜", "王芳", "刘洋", "陈静"]
TEAMS = ["platform", "backend", "sre"]

ALERT_MESSAGES = {
    "critical": ["Pod持续CrashLoopBackOff", "服务完全不可用", "数据库连接池耗尽", "OOMKilled超过3次"],
    "warning": ["CPU使用率超过80%", "内存使用率持续偏高", "请求延迟P99超过2秒", "副本数低于预期"],
    "info": ["自动扩容触发", "配置已更新", "滚动更新完成"],
}

def seed():
    init_db()  # 确保表结构存在
    db = SessionLocal()

    # 如果已经有数据了,先清空,避免重复跑脚本导致数据越堆越多
    db.query(Alert).delete()
    db.query(Deployment).delete()
    db.query(OnCall).delete()
    db.query(Service).delete()
    db.commit()

    now = datetime.datetime.utcnow()

    # 1. 创建服务
    services = []
    for name in SERVICE_NAMES:
        service = Service(
            name=name,
            workload_type=random.choice(["Deployment", "Deployment", "StatefulSet"]),
            namespace=random.choice(NAMESPACES),
            replicas=random.choice([2, 3, 3, 5]),
            status=random.choice(["running", "running", "running", "degraded"]),
        )
        services.append(service)
        db.add(service)
    db.commit()

    # 2. 给每个服务生成0-5条告警记录,时间分布在过去30天内
    for service in services:
        num_alerts = random.randint(0, 5)
        for _ in range(num_alerts):
            severity = random.choice(["critical", "warning", "warning", "info"])
            days_ago = random.randint(0, 30)
            alert = Alert(
                service_id=service.id,
                severity=severity,
                message=random.choice(ALERT_MESSAGES[severity]),
                status=random.choice(["firing", "resolved", "resolved"]),
                triggered_at=now - datetime.timedelta(days=days_ago, hours=random.randint(0, 23)),
            )
            db.add(alert)

    # 3. 给每个服务生成1-4条部署记录
    for service in services:
        num_deployments = random.randint(1, 4)
        for i in range(num_deployments):
            days_ago = random.randint(0, 60)
            deployment = Deployment(
                service_id=service.id,
                version=f"v1.{random.randint(0, 9)}.{random.randint(0, 9)}",
                operator=random.choice(ENGINEERS),
                deployed_at=now - datetime.timedelta(days=days_ago),
                result=random.choice(["success", "success", "success", "failed", "rollback"]),
            )
            db.add(deployment)

    # 4. 生成未来2周的值班表,每3天轮换一次
    for i in range(5):
        start = now + datetime.timedelta(days=i * 3)
        oncall = OnCall(
            engineer=random.choice(ENGINEERS),
            team=random.choice(TEAMS),
            shift_start=start,
            shift_end=start + datetime.timedelta(days=3),
        )
        db.add(oncall)

    db.commit()

    # 打印汇总,确认数据量级
    print(f"已生成 {db.query(Service).count()} 个服务")
    print(f"已生成 {db.query(Alert).count()} 条告警记录")
    print(f"已生成 {db.query(Deployment).count()} 条部署记录")
    print(f"已生成 {db.query(OnCall).count()} 条值班记录")

    db.close()

if __name__ == "__main__":
    seed()