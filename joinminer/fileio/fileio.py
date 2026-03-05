"""
文件I/O管理器

提供统一的多文件后端管理，根据URI scheme自动路由到对应的文件后端实例。
支持在同一个应用中操作多个文件后端（如从HDFS读取，写入S3或本地）。
"""

import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from joinminer.fileio.backends.base import FileBackend
from joinminer.fileio.backends.local import LocalBackend
from joinminer.fileio.backends.hdfs import HDFSBackend

logger = logging.getLogger(__name__)


class FileIO:
    """
    文件I/O管理器

    根据路径的URI scheme (hdfs://, s3://, file://) 自动选择对应的文件后端实例。
    支持在同一个Spark作业中操作多个不同的文件后端。

    使用示例:
        # 配置多个文件后端
        backend_configs = {
            'hdfs': {'host': 'default', 'port': 9000},
            'local': {},
            's3': {'access_key': 'xxx', 'secret_key': 'yyy'}
        }

        # 创建管理器
        fileio = FileIO(backend_configs)

        # 根据路径自动选择文件后端
        hdfs_backend = fileio.get_backend('hdfs:///data/table')
        local_backend = fileio.get_backend('file:///local/path')
        s3_backend = fileio.get_backend('s3://bucket/path')
    """

    # 支持的scheme到文件后端类的映射
    BACKEND_CLASSES = {
        'file': LocalBackend,
        'hdfs': HDFSBackend,
        # 's3': S3Backend,  # 待实现
        # 's3a': S3Backend,  # 待实现
    }

    # scheme的别名映射
    SCHEME_ALIASES = {
        's3a': 's3',
        'local': 'file',
    }

    def __init__(self, backend_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        初始化文件I/O管理器

        Args:
            backend_configs: 文件后端配置字典，格式为:
                {
                    'hdfs': {'host': 'default', 'port': 9000},
                    'local': {},
                    's3': {'access_key': '...', 'secret_key': '...'}
                }
        """
        self.backends: Dict[str, FileBackend] = {}
        self._init_backends(backend_configs or {})

    def _init_backends(self, backend_configs: Dict[str, Dict[str, Any]]) -> None:
        """
        根据配置初始化文件后端实例

        Args:
            backend_configs: 文件后端配置字典
        """
        for scheme, config in backend_configs.items():
            # 处理别名
            normalized_scheme = self.SCHEME_ALIASES.get(scheme, scheme)

            # 获取对应的文件后端类
            backend_class = self.BACKEND_CLASSES.get(normalized_scheme)
            if backend_class is None:
                logger.warning(f"Unknown backend scheme: {scheme}, skipping")
                continue

            # 创建文件后端实例
            try:
                backend_instance = backend_class(**config)
                self.backends[normalized_scheme] = backend_instance
                logger.info(f"Initialized backend for scheme: {normalized_scheme}")
            except Exception as e:
                logger.error(f"Failed to initialize backend for {scheme}: {e}")

    def get_backend(self, path: str) -> FileBackend:
        """
        根据路径的URI scheme返回对应的文件后端实例

        Args:
            path: 文件路径，可以包含scheme (如 hdfs:///path, s3://bucket/path)
                 或不包含scheme (默认为file://)

        Returns:
            对应的FileBackend实例

        Raises:
            ValueError: 如果scheme不支持或未配置

        Examples:
            >>> backend = fileio.get_backend('hdfs:///data/table')
            >>> backend = fileio.get_backend('s3://bucket/data')
            >>> backend = fileio.get_backend('/local/path')  # 默认为file://
        """
        scheme = self._parse_scheme(path)

        # 处理别名
        normalized_scheme = self.SCHEME_ALIASES.get(scheme, scheme)

        # 查找文件后端实例
        if normalized_scheme not in self.backends:
            # 尝试自动创建默认实例
            if normalized_scheme in self.BACKEND_CLASSES:
                logger.info(f"Auto-creating default backend for scheme: {normalized_scheme}")
                backend_class = self.BACKEND_CLASSES[normalized_scheme]
                try:
                    # 使用默认参数创建实例
                    self.backends[normalized_scheme] = backend_class()
                except Exception as e:
                    raise ValueError(
                        f"No backend configured for scheme '{normalized_scheme}' "
                        f"and auto-creation failed: {e}"
                    )
            else:
                raise ValueError(
                    f"No backend configured for scheme: {normalized_scheme}. "
                    f"Available schemes: {list(self.backends.keys())}"
                )

        return self.backends[normalized_scheme]

    def register_backend(self, scheme: str, backend: FileBackend) -> None:
        """
        手动注册一个文件后端实例

        Args:
            scheme: URI scheme (如 'hdfs', 's3', 'file')
            backend: FileBackend实例

        Examples:
            >>> custom_backend = HDFSBackend(host='namenode1', port=8020)
            >>> fileio.register_backend('hdfs', custom_backend)
        """
        # 处理别名
        normalized_scheme = self.SCHEME_ALIASES.get(scheme, scheme)

        self.backends[normalized_scheme] = backend
        logger.info(f"Registered backend for scheme: {normalized_scheme}")

    def has_backend(self, scheme: str) -> bool:
        """
        检查是否配置了指定scheme的文件后端

        Args:
            scheme: URI scheme

        Returns:
            是否已配置
        """
        normalized_scheme = self.SCHEME_ALIASES.get(scheme, scheme)
        return normalized_scheme in self.backends

    def list_schemes(self) -> list:
        """
        列出所有已配置的scheme

        Returns:
            scheme列表
        """
        return list(self.backends.keys())

    def _parse_scheme(self, path: str) -> str:
        """
        解析路径的URI scheme，要求所有路径必须显式指定scheme

        Args:
            path: 必须包含URI scheme的文件路径

        Returns:
            scheme (如 'hdfs', 's3', 'file')

        Raises:
            ValueError: 如果路径缺少URI scheme

        Examples:
            >>> fileio._parse_scheme('hdfs:///data/table')
            'hdfs'
            >>> fileio._parse_scheme('s3://bucket/path')
            's3'
            >>> fileio._parse_scheme('file:///local/path')
            'file'
            >>> fileio._parse_scheme('/local/path')
            ValueError: Path '/local/path' is missing URI scheme...
        """
        # 使用urlparse解析URI
        parsed = urlparse(path)

        if not parsed.scheme:
            raise ValueError(
                f"Path '{path}' is missing URI scheme. "
                f"All paths must have explicit schemes to avoid ambiguity between local and distributed filesystems. "
                f"Examples:\n"
                f"  - Local filesystem: 'file://{path}'\n"
                f"  - HDFS: 'hdfs://{path}'\n"
                f"  - S3: 's3://bucket{path if path.startswith('/') else '/' + path}'"
            )

        return parsed.scheme

    def remove_scheme(self, path: str) -> str:
        """
        从路径中移除URI scheme，返回纯路径

        Args:
            path: 完整路径（可能包含scheme）

        Returns:
            不包含scheme的路径

        Examples:
            >>> fileio.remove_scheme('hdfs:///data/table')
            '/data/table'
            >>> fileio.remove_scheme('s3://bucket/path/file')
            'bucket/path/file' (S3路径不以/开头)
            >>> fileio.remove_scheme('/local/path')
            '/local/path'
        """
        parsed = urlparse(path)

        if parsed.scheme:
            # 对于有scheme的路径
            if parsed.scheme == 'file':
                # file:// scheme，返回path部分
                return parsed.path
            elif parsed.scheme in ['hdfs']:
                # hdfs:// scheme，返回path部分
                return parsed.path
            elif parsed.scheme in ['s3', 's3a']:
                # s3:// scheme，返回netloc + path
                return f"{parsed.netloc}{parsed.path}"
            else:
                # 其他scheme，返回netloc + path
                return f"{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
        else:
            # 没有scheme，直接返回原路径
            return path

    # ========================================
    # URI-aware wrapper methods
    # These methods accept paths with URI schemes and handle routing internally
    # ========================================

    def exists(self, path: str) -> bool:
        """
        检查路径是否存在 (URI-aware)

        Args:
            path: 文件或目录路径，可以包含URI scheme
                 Examples: 'hdfs:///data/table', '/local/path', 's3://bucket/key'

        Returns:
            如果路径存在返回True，否则返回False

        Examples:
            >>> fileio.exists('hdfs:///data/table')  # Auto-routes to HDFS
            True
            >>> fileio.exists('/local/path')          # Auto-routes to Local
            False
        """
        backend = self.get_backend(path)
        clean_path = self.remove_scheme(path)
        return backend.exists(clean_path)

    def mkdir(self, path: str, parents: bool = True) -> None:
        """
        创建目录 (URI-aware)

        Args:
            path: 目录路径，可以包含URI scheme
            parents: 是否创建父目录（类似 mkdir -p）

        Examples:
            >>> fileio.mkdir('hdfs:///data/new_table', parents=True)
            >>> fileio.mkdir('file:///tmp/test')
        """
        backend = self.get_backend(path)
        clean_path = self.remove_scheme(path)
        backend.mkdir(clean_path, parents=parents)

    def list(self, path: str, recursive: bool = False) -> List[str]:
        """
        列出目录内容 (URI-aware)

        Args:
            path: 目录路径，必须包含URI scheme
            recursive: 是否递归列出子目录

        Returns:
            文件和目录路径列表，保留原始URI scheme

        Raises:
            ValueError: 如果路径缺少URI scheme

        Examples:
            >>> fileio.list('hdfs:///data')
            ['hdfs:///data/table1', 'hdfs:///data/table2']
            >>> fileio.list('file:///local/dir')
            ['file:///local/dir/file1.txt', 'file:///local/dir/file2.txt']
        """
        backend = self.get_backend(path)
        clean_path = self.remove_scheme(path)
        scheme = self._parse_scheme(path)

        # Get paths from backend
        paths = backend.list(clean_path, recursive=recursive)

        # Add scheme prefix back to all paths
        scheme_prefix = f"{scheme}://"
        paths = [
            f"{scheme_prefix}{p}" if not p.startswith(scheme_prefix) else p
            for p in paths
        ]

        return paths

    def delete(self, path: str, recursive: bool = False, missing_ok: bool = True) -> bool:
        """
        删除文件或目录 (URI-aware)

        Args:
            path: 文件或目录路径，可以包含URI scheme
            recursive: 是否递归删除目录内容
            missing_ok: 路径不存在时是否静默处理（默认True）

        Returns:
            True if something was deleted, False if path didn't exist

        Raises:
            FileNotFoundError: 当 missing_ok=False 且路径不存在时
            OSError: 删除操作失败时（权限问题、目录非空等）

        Examples:
            >>> fileio.delete('hdfs:///data/old_table', recursive=True)
            True
            >>> fileio.delete('file:///tmp/not_exists.txt')  # missing_ok=True by default
            False
            >>> fileio.delete('file:///tmp/not_exists.txt', missing_ok=False)  # raises FileNotFoundError
        """
        backend = self.get_backend(path)
        clean_path = self.remove_scheme(path)
        return backend.delete(clean_path, recursive=recursive, missing_ok=missing_ok)

    def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        """
        读取文本文件 (URI-aware)

        Args:
            path: 文件路径，可以包含URI scheme
            encoding: 文件编码

        Returns:
            文件内容字符串

        Examples:
            >>> content = fileio.read_text('hdfs:///config.txt')
            >>> content = fileio.read_text('/local/file.txt')
        """
        backend = self.get_backend(path)
        clean_path = self.remove_scheme(path)
        return backend.read_text(clean_path, encoding=encoding)

    def write_text(self, path: str, content: str, encoding: str = 'utf-8') -> None:
        """
        写入文本文件 (URI-aware)

        Args:
            path: 文件路径，可以包含URI scheme
            content: 要写入的内容
            encoding: 文件编码

        Examples:
            >>> fileio.write_text('hdfs:///output.txt', 'Hello World')
            >>> fileio.write_text('/local/log.txt', 'Log entry')
        """
        backend = self.get_backend(path)
        clean_path = self.remove_scheme(path)
        backend.write_text(clean_path, content, encoding=encoding)

    def read_json(self, path: str) -> Dict[str, Any]:
        """
        读取JSON文件 (URI-aware)

        Args:
            path: JSON文件路径，可以包含URI scheme

        Returns:
            解析后的字典对象

        Examples:
            >>> data = fileio.read_json('hdfs:///config.json')
            >>> data = fileio.read_json('/local/settings.json')
        """
        backend = self.get_backend(path)
        clean_path = self.remove_scheme(path)
        return backend.read_json(clean_path)

    def write_json(self, path: str, data: Dict[str, Any], indent: int = 2) -> None:
        """
        写入JSON文件 (URI-aware)

        Args:
            path: JSON文件路径，可以包含URI scheme
            data: 要写入的字典对象
            indent: JSON缩进空格数

        Examples:
            >>> fileio.write_json('hdfs:///output.json', {'key': 'value'})
            >>> fileio.write_json('/local/data.json', data_dict)
        """
        backend = self.get_backend(path)
        clean_path = self.remove_scheme(path)
        backend.write_json(clean_path, data, indent=indent)

    def read_yaml(self, path: str) -> Dict[str, Any]:
        """
        读取YAML文件 (URI-aware)

        Args:
            path: YAML文件路径，可以包含URI scheme

        Returns:
            解析后的字典对象

        Examples:
            >>> config = fileio.read_yaml('hdfs:///config.yaml')
            >>> config = fileio.read_yaml('/local/config.yml')
        """
        backend = self.get_backend(path)
        clean_path = self.remove_scheme(path)
        return backend.read_yaml(clean_path)

    def write_yaml(self, path: str, data: Dict[str, Any]) -> None:
        """
        写入YAML文件 (URI-aware)

        Args:
            path: YAML文件路径，可以包含URI scheme
            data: 要写入的字典对象

        Examples:
            >>> fileio.write_yaml('hdfs:///config.yaml', config_dict)
            >>> fileio.write_yaml('/local/settings.yml', settings)
        """
        backend = self.get_backend(path)
        clean_path = self.remove_scheme(path)
        backend.write_yaml(clean_path, data)
