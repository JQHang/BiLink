"""
本地文件后端实现
"""

import os
import json
import yaml
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any

from joinminer.fileio.backends.base import FileBackend

logger = logging.getLogger(__name__)


class LocalBackend(FileBackend):
    """本地文件后端实现"""

    def exists(self, path: str) -> bool:
        """检查路径是否存在"""
        return os.path.exists(path)

    def mkdir(self, path: str, parents: bool = True) -> None:
        """创建目录"""
        if parents:
            os.makedirs(path, exist_ok=True)
        else:
            os.mkdir(path)

    def list(self, path: str, recursive: bool = False) -> List[str]:
        """
        列出目录内容

        Args:
            path: 目录路径
            recursive: 是否递归列出子目录

        Returns:
            文件和目录的完整路径列表
        """
        if not recursive:
            # 非递归：只列出直接子文件和目录
            items = os.listdir(path)
            return [os.path.join(path, item) for item in items]
        else:
            # 递归：列出所有文件
            result = []
            for root, dirs, files in os.walk(path):
                for name in files:
                    result.append(os.path.join(root, name))
            return result

    def delete(self, path: str, recursive: bool = False, missing_ok: bool = True) -> bool:
        """删除文件或目录"""
        if not self.exists(path):
            if not missing_ok:
                raise FileNotFoundError(f"Path does not exist: {path}")
            logger.debug(f"Path does not exist, skipping: {path}")
            return False

        if os.path.isfile(path):
            os.remove(path)
            logger.debug(f"Deleted file: {path}")
        elif os.path.isdir(path):
            if recursive:
                shutil.rmtree(path)
                logger.debug(f"Recursively deleted directory: {path}")
            else:
                # os.rmdir() 只能删除空目录，如果目录非空会抛出 OSError
                # 先检查目录是否为空，提供更清晰的错误消息
                if os.listdir(path):
                    raise OSError(f"Directory not empty (use recursive=True to force): {path}")
                os.rmdir(path)
                logger.debug(f"Deleted empty directory: {path}")

        return True

    def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        """读取文本文件"""
        with open(path, 'r', encoding=encoding) as f:
            return f.read()

    def write_text(self, path: str, content: str, encoding: str = 'utf-8') -> None:
        """写入文本文件"""
        # 确保目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            self.mkdir(dir_path, parents=True)

        with open(path, 'w', encoding=encoding) as f:
            f.write(content)

    def read_json(self, path: str) -> Dict[str, Any]:
        """读取JSON文件"""
        content = self.read_text(path)
        return json.loads(content)

    def write_json(self, path: str, data: Dict[str, Any], indent: int = 2) -> None:
        """写入JSON文件"""
        json_str = json.dumps(data, indent=indent, ensure_ascii=False)
        self.write_text(path, json_str)

    def read_yaml(self, path: str) -> Dict[str, Any]:
        """读取YAML文件"""
        content = self.read_text(path)
        return yaml.safe_load(content)

    def write_yaml(self, path: str, data: Dict[str, Any]) -> None:
        """写入YAML文件"""
        yaml_str = yaml.dump(data,
                            default_flow_style=False,  # 使用块样式格式
                            indent=2,                   # 2个空格缩进
                            sort_keys=False,            # 保持原始顺序
                            allow_unicode=True)         # 支持Unicode字符
        self.write_text(path, yaml_str)
