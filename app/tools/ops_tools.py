import random
import datetime

def restart_service(service_name: str) -> dict:
    """模拟重启一个服务"""
    return {
        "tool": "restart_service",
        "service": service_name,
        "status": "success",
        "message": f"服务 {service_name} 已成功重启,所有副本已恢复 Running 状态",
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

def get_logs(service_name: str, lines: int = 20) -> dict:
    """模拟获取一个服务最近的日志"""
    sample_logs = [
        f"[INFO] {service_name} handling request /api/v1/health",
        f"[INFO] {service_name} connected to database pool",
        f"[WARN] {service_name} request latency 850ms exceeds threshold",
        f"[INFO] {service_name} cache hit ratio: 92%",
        f"[ERROR] {service_name} failed to connect to downstream service, retrying...",
    ]
    logs = [random.choice(sample_logs) for _ in range(min(lines, 10))]
    return {
        "tool": "get_logs",
        "service": service_name,
        "lines_requested": lines,
        "logs": logs,
    }

def get_resource_usage(service_name: str) -> dict:
    """模拟查看一个服务当前的资源占用情况"""
    return {
        "tool": "get_resource_usage",
        "service": service_name,
        "cpu_usage_percent": round(random.uniform(20, 95), 1),
        "memory_usage_percent": round(random.uniform(30, 90), 1),
        "replicas_running": random.randint(1, 5),
    }

def scale_service(service_name: str, replicas: int) -> dict:
    """模拟将一个服务扩容或缩容到指定副本数"""
    if replicas < 1 or replicas > 20:
        return {
            "tool": "scale_service",
            "service": service_name,
            "status": "rejected",
            "message": f"副本数 {replicas} 超出安全范围(1-20),已拒绝执行",
        }
    return {
        "tool": "scale_service",
        "service": service_name,
        "status": "success",
        "message": f"服务 {service_name} 已扩缩容至 {replicas} 个副本",
        "replicas": replicas,
    }