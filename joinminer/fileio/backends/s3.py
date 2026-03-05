"""
S3文件后端实现

基于s3fs库实现S3兼容存储的文件后端接口。
支持AWS S3、MinIO等S3兼容的对象存储。

注意：使用此模块需要安装s3fs库:
    pip install s3fs
"""

import logging
from typing import List, Dict, Any, Optional

from joinminer.fileio.backends.base import FileBackend

logger = logging.getLogger(__name__)


class S3Backend(FileBackend):
    """
    S3文件后端实现

    基于s3fs库，支持AWS S3和S3兼容存储（如MinIO、阿里云OSS等）

    使用示例:
        # AWS S3
        s3_backend = S3Backend(
            access_key='your-access-key',
            secret_key='your-secret-key',
            region='us-west-2'
        )

        # MinIO
        minio_backend = S3Backend(
            access_key='minioadmin',
            secret_key='minioadmin',
            endpoint_url='http://localhost:9000'
        )

        # 读写操作
        s3_backend.write_text('bucket/path/file.txt', 'content')
        content = s3_backend.read_text('bucket/path/file.txt')
    """

    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = 'us-east-1',
        endpoint_url: Optional[str] = None,
        **kwargs
    ):
        """
        初始化S3文件后端

        Args:
            access_key: AWS Access Key ID (可选，也可从环境变量读取)
            secret_key: AWS Secret Access Key (可选，也可从环境变量读取)
            region: AWS区域
            endpoint_url: S3兼容存储的endpoint (用于MinIO等)
            **kwargs: 其他s3fs参数
        """
        try:
            import s3fs
        except ImportError:
            raise ImportError(
                "S3Backend requires s3fs library. "
                "Install it with: pip install s3fs"
            )

        # 构建s3fs配置
        s3fs_config = {
            'key': access_key,
            'secret': secret_key,
            'client_kwargs': {'region_name': region}
        }

        if endpoint_url:
            s3fs_config['client_kwargs']['endpoint_url'] = endpoint_url

        # 合并额外参数
        s3fs_config.update(kwargs)

        # 创建s3fs实例
        self.fs = s3fs.S3FileSystem(**s3fs_config)
        self.region = region
        self.endpoint_url = endpoint_url

        logger.info(f"Initialized S3Backend (region={region}, endpoint={endpoint_url})")

    def exists(self, path: str) -> bool:
        """检查路径是否存在"""
        try:
            return self.fs.exists(path)
        except Exception as e:
            logger.error(f"Error checking S3 path existence {path}: {e}")
            return False

    def mkdir(self, path: str, parents: bool = True) -> None:
        """
        创建目录（S3中实际不需要创建目录）

        S3是对象存储，没有真正的目录概念。
        此方法为了接口一致性保留，但实际不执行操作。
        """
        # S3不需要创建目录
        logger.debug(f"S3 mkdir (no-op): {path}")
        pass

    def list(self, path: str, recursive: bool = False) -> List[str]:
        """
        列出目录内容

        Args:
            path: S3路径 (格式: bucket/prefix/)
            recursive: 是否递归列出

        Returns:
            文件路径列表
        """
        try:
            if recursive:
                return self.fs.find(path)
            else:
                return self.fs.ls(path)
        except Exception as e:
            logger.error(f"Error listing S3 path {path}: {e}")
            return []

    def delete(self, path: str, recursive: bool = False, missing_ok: bool = True) -> bool:
        """删除文件或目录"""
        if not self.fs.exists(path):
            if not missing_ok:
                raise FileNotFoundError(f"S3 path does not exist: {path}")
            logger.debug(f"S3 path does not exist, skipping: {path}")
            return False

        # s3fs.rm() 已经处理了 recursive 逻辑
        self.fs.rm(path, recursive=recursive)
        logger.debug(f"Deleted S3 path: {path}{' (recursive)' if recursive else ''}")
        return True

    def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        """读取文本文件"""
        try:
            with self.fs.open(path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read S3 file {path}: {e}")

    def write_text(self, path: str, content: str, encoding: str = 'utf-8') -> None:
        """写入文本文件"""
        try:
            with self.fs.open(path, 'w', encoding=encoding) as f:
                f.write(content)
            logger.info(f"Written to S3: {path}")
        except Exception as e:
            raise ValueError(f"Failed to write S3 file {path}: {e}")

    def read_json(self, path: str) -> Dict[str, Any]:
        """读取JSON文件"""
        import json
        try:
            content = self.read_text(path)
            return json.loads(content)
        except Exception as e:
            raise ValueError(f"Failed to read JSON from S3 {path}: {e}")

    def write_json(self, path: str, data: Dict[str, Any], indent: int = 2) -> None:
        """写入JSON文件"""
        import json
        try:
            json_str = json.dumps(data, indent=indent, ensure_ascii=False)
            self.write_text(path, json_str)
        except Exception as e:
            raise ValueError(f"Failed to write JSON to S3 {path}: {e}")

    def read_yaml(self, path: str) -> Dict[str, Any]:
        """读取YAML文件"""
        import yaml
        try:
            content = self.read_text(path)
            return yaml.safe_load(content)
        except Exception as e:
            raise ValueError(f"Failed to read YAML from S3 {path}: {e}")

    def write_yaml(self, path: str, data: Dict[str, Any]) -> None:
        """写入YAML文件"""
        import yaml
        try:
            yaml_str = yaml.dump(
                data,
                default_flow_style=False,
                indent=2,
                sort_keys=False,
                allow_unicode=True
            )
            self.write_text(path, yaml_str)
        except Exception as e:
            raise ValueError(f"Failed to write YAML to S3 {path}: {e}")
