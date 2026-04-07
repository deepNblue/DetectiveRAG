"""
认证系统
提供基本的用户认证功能
"""

from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from loguru import logger
from datetime import datetime


class AuthSystem:
    """
    认证系统
    支持基本的用户名/密码认证
    """
    
    def __init__(self, db_path: str = "./data/auth.db"):
        """
        初始化认证系统
        
        Args:
            db_path: SQLite数据库路径
        """
        self.auth = HTTPBasicAuth()
        self.db_path = db_path
        self.logger = logger.bind(module="auth")
        
        # 确保数据库目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_db()
        
        # 创建演示账号
        self._create_demo_user()
        
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # 创建登录日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS login_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                success BOOLEAN
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"认证数据库初始化完成: {self.db_path}")
    
    def _create_demo_user(self):
        """创建演示用户"""
        demo_username = "demo"
        demo_password = "REDACTED_DEMO_PASSWORD"
        
        if not self.user_exists(demo_username):
            self.create_user(demo_username, demo_password)
            self.logger.info(f"演示账号创建成功: {demo_username}")
    
    def create_user(self, username: str, password: str) -> bool:
        """
        创建用户
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            是否创建成功
        """
        try:
            password_hash = generate_password_hash(password)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash)
            )
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"用户创建成功: {username}")
            return True
            
        except sqlite3.IntegrityError:
            self.logger.warning(f"用户已存在: {username}")
            return False
        except Exception as e:
            self.logger.error(f"用户创建失败: {e}")
            return False
    
    def user_exists(self, username: str) -> bool:
        """检查用户是否存在"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
    
    def verify_password(self, username: str, password: str) -> bool:
        """
        验证密码
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            是否验证成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (username,)
        )
        result = cursor.fetchone()
        
        if result:
            password_hash = result[0]
            success = check_password_hash(password_hash, password)
            
            # 记录登录日志
            self._log_login(username, success)
            
            # 更新最后登录时间
            if success:
                cursor.execute(
                    "UPDATE users SET last_login = ? WHERE username = ?",
                    (datetime.now(), username)
                )
                conn.commit()
            
            conn.close()
            return success
        
        conn.close()
        return False
    
    def _log_login(self, username: str, success: bool):
        """记录登录日志"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO login_logs (username, success) VALUES (?, ?)",
                (username, success)
            )
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"登录日志记录失败: {e}")
    
    def get_user_info(self, username: str) -> dict:
        """获取用户信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT username, created_at, last_login FROM users WHERE username = ?",
            (username,)
        )
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return {
                "username": result[0],
                "created_at": result[1],
                "last_login": result[2]
            }
        return None
    
    def get_login_history(self, username: str, limit: int = 10) -> list:
        """获取登录历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT login_time, ip_address, success 
            FROM login_logs 
            WHERE username = ? 
            ORDER BY login_time DESC 
            LIMIT ?
            """,
            (username, limit)
        )
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "login_time": row[0],
                "ip_address": row[1],
                "success": bool(row[2])
            }
            for row in results
        ]
