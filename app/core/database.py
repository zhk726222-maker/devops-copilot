from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import datetime

# SQLite数据库文件存放路径,会自动在data文件夹下生成一个db文件
DATABASE_URL = "sqlite:///data/devops.db"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Service(Base):
    """服务表:记录有哪些服务在跑"""
    __tablename__ = "services"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)       # 服务名,如 payment-api
    workload_type = Column(String, nullable=False)            # Deployment / StatefulSet
    namespace = Column(String, nullable=False)                 # 所属命名空间
    replicas = Column(Integer, nullable=False, default=1)      # 副本数
    status = Column(String, nullable=False, default="running") # running / degraded / down

    alerts = relationship("Alert", back_populates="service")
    deployments = relationship("Deployment", back_populates="service")


class Alert(Base):
    """告警表:记录历史告警"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    service_id = Column(Integer, ForeignKey("services.id"), nullable=False)
    severity = Column(String, nullable=False)      # critical / warning / info
    message = Column(String, nullable=False)
    status = Column(String, nullable=False, default="firing")  # firing / resolved
    triggered_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)

    service = relationship("Service", back_populates="alerts")


class Deployment(Base):
    """部署记录表:记录每次发布"""
    __tablename__ = "deployments"

    id = Column(Integer, primary_key=True)
    service_id = Column(Integer, ForeignKey("services.id"), nullable=False)
    version = Column(String, nullable=False)
    operator = Column(String, nullable=False)        # 操作人
    deployed_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    result = Column(String, nullable=False, default="success")  # success / failed / rollback

    service = relationship("Service", back_populates="deployments")


class OnCall(Base):
    """值班表"""
    __tablename__ = "oncall"

    id = Column(Integer, primary_key=True)
    engineer = Column(String, nullable=False)
    team = Column(String, nullable=False)
    shift_start = Column(DateTime, nullable=False)
    shift_end = Column(DateTime, nullable=False)


def init_db():
    """根据上面定义的表结构,在数据库里真正创建这些表(如果还不存在的话)"""
    Base.metadata.create_all(bind=engine)
    print("数据库表结构已创建")


if __name__ == "__main__":
    init_db()