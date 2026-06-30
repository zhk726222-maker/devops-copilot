from app.core.config import client, DEFAULT_MODEL
from app.core.database import SessionLocal
from sqlalchemy import text

# 把表结构写成清晰的文字描述,喂给大模型作为生成SQL的依据
SCHEMA_DESCRIPTION = """
数据库包含以下4张表:

1. services(服务表)
   - id: 主键
   - name: 服务名(如 payment-api)
   - workload_type: 工作负载类型(Deployment / StatefulSet)
   - namespace: 命名空间(production / staging)
   - replicas: 副本数
   - status: 当前状态(running / degraded / down)

2. alerts(告警表)
   - id: 主键
   - service_id: 外键,关联 services.id
   - severity: 告警级别(critical / warning / info)
   - message: 告警内容
   - status: 告警状态(firing 触发中 / resolved 已解决)
   - triggered_at: 触发时间

3. deployments(部署记录表)
   - id: 主键
   - service_id: 外键,关联 services.id
   - version: 版本号
   - operator: 操作人
   - deployed_at: 部署时间
   - result: 部署结果(success / failed / rollback)

4. oncall(值班表)
   - id: 主键
   - engineer: 值班工程师
   - team: 所属团队
   - shift_start: 值班开始时间
   - shift_end: 值班结束时间
"""

SQL_GEN_PROMPT = """你是一个SQL生成助手,任务是把用户的自然语言问题转换成一条可执行的SQLite查询语句。

数据库表结构如下:
{schema}

要求:
1. 只生成SELECT查询语句,不允许生成INSERT、UPDATE、DELETE、DROP等修改数据的语句
2. 只返回SQL语句本身,不要任何解释文字,不要用markdown代码块包裹
3. 涉及多表关联时使用JOIN
4. 时间相关的查询,当前时间可以用 datetime('now') 表示

用户问题:{query}

SQL语句:"""

def generate_sql(query: str) -> str:
    """调用大模型,把自然语言问题转成SQL语句"""
    prompt = SQL_GEN_PROMPT.format(schema=SCHEMA_DESCRIPTION, query=query)
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    sql = response.choices[0].message.content.strip()
    # 防御性清理:有些模型即使被要求不要markdown包裹,仍可能习惯性加上```sql代码块,这里做兜底清理
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql

def is_safe_sql(sql: str) -> bool:
    """安全校验:只允许SELECT查询,拦截任何可能修改数据的语句"""
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return False
    forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"]
    return not any(keyword in sql_upper for keyword in forbidden_keywords)

def execute_sql(sql: str) -> list[dict]:
    """执行SQL并返回查询结果,执行前会先做安全校验"""
    if not is_safe_sql(sql):
        raise ValueError(f"检测到不安全的SQL语句,已拦截执行: {sql}")

    db = SessionLocal()
    try:
        result = db.execute(text(sql))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
        return rows
    finally:
        db.close()

if __name__ == "__main__":
    test_query = "过去7天里哪些服务出现过critical级别的告警,把服务名和告警内容列出来"
    sql = generate_sql(test_query)
    print(f"用户问题: {test_query}")
    print(f"生成的SQL:\n{sql}\n")

    results = execute_sql(sql)
    print(f"查询结果({len(results)}条):")
    for row in results:
        print(row)