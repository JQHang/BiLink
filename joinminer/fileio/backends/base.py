"""
文件后端抽象基类

定义统一的文件后端接口，支持本地文件系统、HDFS等多种后端
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class FileBackend(ABC):
    """文件后端基础接口 - 只包含通用文件操作"""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """
        检查路径是否存在

        Args:
            path: 文件或目录路径

        Returns:
            如果路径存在返回True，否则返回False
        """
        pass

    @abstractmethod
    def mkdir(self, path: str, parents: bool = True) -> None:
        """
        创建目录

        Args:
            path: 目录路径
            parents: 是否创建父目录（类似 mkdir -p）
        """
        pass

    @abstractmethod
    def list(self, path: str, recursive: bool = False) -> List[str]:
        """
        列出目录内容

        Args:
            path: 目录路径
            recursive: 是否递归列出子目录

        Returns:
            文件和目录路径列表
        """
        pass

    @abstractmethod
    def delete(self, path: str, recursive: bool = False, missing_ok: bool = True) -> bool:
        """
        删除文件或目录

        Args:
            path: 文件或目录路径
            recursive: 是否递归删除目录内容
            missing_ok: 路径不存在时是否静默处理（默认True）

        Returns:
            True if something was deleted, False if path didn't exist

        Raises:
            FileNotFoundError: 当 missing_ok=False 且路径不存在时
            OSError: 删除操作失败时（权限问题、目录非空等）
        """
        pass

    @abstractmethod
    def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        """
        读取文本文件

        Args:
            path: 文件路径
            encoding: 文件编码

        Returns:
            文件内容字符串
        """
        pass

    @abstractmethod
    def write_text(self, path: str, content: str, encoding: str = 'utf-8') -> None:
        """
        写入文本文件

        Args:
            path: 文件路径
            content: 要写入的内容
            encoding: 文件编码
        """
        pass

    @abstractmethod
    def read_json(self, path: str) -> Dict[str, Any]:
        """
        读取JSON文件

        Args:
            path: JSON文件路径

        Returns:
            解析后的字典对象
        """
        pass

    @abstractmethod
    def write_json(self, path: str, data: Dict[str, Any], indent: int = 2) -> None:
        """
        写入JSON文件

        Args:
            path: JSON文件路径
            data: 要写入的字典对象
            indent: JSON缩进空格数
        """
        pass

    @abstractmethod
    def read_yaml(self, path: str) -> Dict[str, Any]:
        """
        读取YAML文件

        Args:
            path: YAML文件路径

        Returns:
            解析后的字典对象
        """
        pass

    @abstractmethod
    def write_yaml(self, path: str, data: Dict[str, Any]) -> None:
        """
        写入YAML文件

        Args:
            path: YAML文件路径
            data: 要写入的字典对象
        """
        pass
