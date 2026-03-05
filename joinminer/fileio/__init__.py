"""
文件I/O模块

提供统一的文件读写接口，支持多种文件后端（本地、HDFS、S3等）。
主要用于读写配置文件、标记文件等小文件，配合Spark进行数据处理。

使用示例:
    from joinminer.fileio import FileIO
    from joinminer.fileio.backends import LocalBackend, HDFSBackend

    # 创建FileIO管理器
    fileio = FileIO({
        'local': {},
        'hdfs': {'host': 'default', 'port': 9000}
    })

    # URI-aware读写
    config = fileio.read_yaml('file:///path/to/config.yaml')
    fileio.write_json('hdfs:///path/to/output.json', data)

    # 或直接使用特定后端
    local_backend = LocalBackend()
    local_backend.write_text('/path/to/file.txt', 'content')
"""

from joinminer.fileio.fileio import FileIO
from joinminer.fileio import backends

__all__ = [
    'FileIO',
    'backends',
]
