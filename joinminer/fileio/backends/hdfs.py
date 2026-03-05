"""
HDFS文件后端实现
"""

import json
import yaml
import logging
from pyarrow import fs as pafs
from typing import List, Dict, Any

from joinminer.fileio.backends.base import FileBackend

logger = logging.getLogger(__name__)


class HDFSBackend(FileBackend):
    """HDFS文件后端实现，基于PyArrow"""

    def __init__(self, host: str = 'default', port: int = 9000):
        """
        初始化HDFS文件后端

        Args:
            host: HDFS namenode地址
            port: HDFS端口
        """
        self.host = host
        self.port = port
        self.fs = pafs.HadoopFileSystem(host=host, port=port)

    def exists(self, path: str) -> bool:
        """检查路径是否存在"""
        info = self.fs.get_file_info(path)
        return info.type != pafs.FileType.NotFound

    def mkdir(self, path: str, parents: bool = True) -> None:
        """创建目录"""
        self.fs.create_dir(path, recursive=parents)
        logger.info(f'Created directory at {path}')

    def list(self, path: str, recursive: bool = False) -> List[str]:
        """
        列出目录内容

        Args:
            path: 目录路径
            recursive: 是否递归列出子目录

        Returns:
            文件和目录路径列表
        """
        file_info_list = self.fs.get_file_info(pafs.FileSelector(path, recursive=recursive))
        return [info.path for info in file_info_list if info.type != pafs.FileType.NotFound]

    def list_files(self, path: str, recursive: bool = False) -> List[str]:
        """
        列出目录中的文件（不包括目录）

        Args:
            path: 目录路径
            recursive: 是否递归列出

        Returns:
            文件路径列表
        """
        file_info_list = self.fs.get_file_info(pafs.FileSelector(path, recursive=recursive))
        return [info.path for info in file_info_list if info.type == pafs.FileType.File]

    def list_directories(self, path: str, recursive: bool = False) -> List[str]:
        """
        列出目录中的子目录（不包括文件）

        Args:
            path: 目录路径
            recursive: 是否递归列出

        Returns:
            目录路径列表
        """
        file_info_list = self.fs.get_file_info(pafs.FileSelector(path, recursive=recursive))
        return [info.path for info in file_info_list if info.type == pafs.FileType.Directory]

    def delete(self, path: str, recursive: bool = False, missing_ok: bool = True) -> bool:
        """删除文件或目录"""
        if not self.exists(path):
            if not missing_ok:
                raise FileNotFoundError(f"Path does not exist: {path}")
            logger.debug(f"Path does not exist, skipping: {path}")
            return False

        info = self.fs.get_file_info(path)

        if info.type == pafs.FileType.File:
            self.fs.delete_file(path)
            logger.debug(f"Deleted HDFS file: {path}")
        else:  # Directory
            if not recursive:
                # 检查目录是否为空
                # PyArrow 的 delete_dir 总是递归删除，所以需要先检查
                contents = self.fs.get_file_info(pafs.FileSelector(path, recursive=False))
                # 过滤掉目录本身，只看内容
                actual_contents = [item for item in contents if item.path != path]
                if actual_contents:
                    raise OSError(f"Directory not empty (use recursive=True to force): {path}")

            self.fs.delete_dir(path)
            logger.debug(f"Deleted HDFS directory: {path}{' (recursive)' if recursive else ''}")

        return True

    def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        """读取文本文件"""
        with self.fs.open_input_stream(path) as f:
            return f.read().decode(encoding)

    def write_text(self, path: str, content: str, encoding: str = 'utf-8') -> None:
        """
        写入文本文件

        Args:
            path: 文件路径
            content: 要写入的内容
            encoding: 文件编码
        """
        with self.fs.open_output_stream(path) as f:
            f.write(content.encode(encoding))

        if len(content) > 500:
            display_content = content[:500] + f"... (total {len(content)} chars)"
        else:
            display_content = content
        logger.info(f'File created at {path} with content: {display_content}')

    def read_json(self, path: str) -> Dict[str, Any]:
        """读取JSON文件"""
        content = self.read_text(path)
        data = json.loads(content)
        logger.info(f'JSON file read successfully from {path}')
        return data

    def write_json(self, path: str, data: Dict[str, Any], indent: int = 2) -> None:
        """写入JSON文件"""
        json_str = json.dumps(data, indent=indent, ensure_ascii=False)
        self.write_text(path, json_str)

    def read_yaml(self, path: str) -> Dict[str, Any]:
        """读取YAML文件"""
        content = self.read_text(path)
        data = yaml.safe_load(content)
        logger.info(f'YAML file read successfully from {path}')
        return data

    def write_yaml(self, path: str, data: Dict[str, Any]) -> None:
        """写入YAML文件"""
        yaml_str = yaml.dump(data,
                            default_flow_style=False,
                            indent=2,
                            sort_keys=False,
                            allow_unicode=True)
        self.write_text(path, yaml_str)
