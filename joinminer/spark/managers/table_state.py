"""
Spark表状态管理器

负责管理Spark表的完成状态标记，通过_COMPLETE标志文件来跟踪表/分区的计算状态。
支持分区表和非分区表的状态管理。

注意：此类仅管理表状态元数据（_COMPLETE标记）。
对于数据清理操作，请使用 joinminer.spark.io.cleanup_table() 函数。
"""

import logging
from typing import List, Tuple, Optional
from datetime import datetime

from joinminer.fileio import FileIO
from joinminer.spark.io.partition import build_partition_path

logger = logging.getLogger(__name__)


class TableStateManager:
    """
    管理Spark表的完成状态（支持分区表和非分区表）

    通过_COMPLETE标志文件追踪表的计算完成状态，支持：
    - 非分区表：检查和标记整个表的完成状态
    - 分区表：检查和标记各分区的完成状态

    接受FileIO实例，支持URI感知的路径处理。
    所有路径参数都可以包含URI scheme（如 hdfs:///data, file:///local）。

    注意：此类仅管理元数据（_COMPLETE标记）。对于数据清理，
    请使用 `from joinminer.spark.io import cleanup_table`。
    """

    def __init__(self, fileio: FileIO, ignore_complete: bool = False):
        """
        初始化表状态管理器

        Args:
            fileio: FileIO实例
                    支持自动URI路由和路径清理
            ignore_complete: 如果为True，check_complete()总是返回(False, [])，
                           即忽略所有完成状态，强制重建所有表。
                           用于实现--force标志。(默认: False)
        """
        self.fileio = fileio
        self.ignore_complete = ignore_complete

    def check_complete(self,
                      base_path: str,
                      partition_columns: Optional[List[str]] = None,
                      partition_instances: Optional[List[List[str]]] = None) -> Tuple[bool, List[List[str]]]:
        """
        检查表或分区是否完成（通过_COMPLETE标志文件）- URI-aware

        Args:
            base_path: 表的基础路径，可以包含URI scheme
                      Examples: 'hdfs:///data/table', 'file:///local/table'
            partition_columns: 分区列名列表，例如 ['date', 'category']
                             非分区表不传此参数（默认None）
            partition_instances: 分区实例列表，例如 [['2024-01-01', 'A'], ['2024-01-01', 'B']]
                               如果为空列表或None，则检查所有可能的分区

        Returns:
            Tuple[bool, List[List[str]]]: (is_complete, missing_partition_instances)
            - is_complete: 是否完成（对非分区表，表示整表是否完成；对分区表，表示所有指定分区是否完成）
            - missing_partition_instances: 缺失的分区实例列表（对非分区表始终为空列表[]）
        """
        # 如果设置了ignore_complete标志，总是返回未完成状态
        if self.ignore_complete:
            logger.info(f"Ignoring completion status for {base_path} (force rebuild enabled)")
            if not partition_columns:
                return False, None
            else:
                return False, partition_instances if partition_instances else []

        # 处理空分区列（非分区表）
        if not partition_columns:
            complete_file = f"{base_path}/_COMPLETE"
            result = self.fileio.exists(complete_file)  # URI-aware!
            logger.info(f"Checked {complete_file}: {'exists' if result else 'missing'}")
            return result, None

        # 如果没有指定要检查的分区，返回True（表示不需要检查）
        if not partition_instances:
            logger.info(f"No partition instances specified for checking, skipping check")
            return True, []

        # 检查每个分区实例
        missing_partition_instances = []
        for partition_instance in partition_instances:
            if len(partition_columns) != len(partition_instance):
                raise ValueError(
                    f"Number of partition columns ({len(partition_columns)}) "
                    f"and values ({len(partition_instance)}) don't match. "
                    f"Columns: {partition_columns}, Values: {partition_instance}"
                )

            # 构建分区路径（保留URI scheme）
            partition_path = build_partition_path(base_path, partition_columns, partition_instance)
            complete_file = f"{partition_path}/_COMPLETE"

            if not self.fileio.exists(complete_file):  # URI-aware!
                missing_partition_instances.append(partition_instance)

        is_complete = len(missing_partition_instances) == 0

        if is_complete:
            logger.info(f"All {len(partition_instances)} partition instances for {partition_columns} in {base_path} are complete")
        else:
            logger.info(f"Missing {len(missing_partition_instances)} partition instances for {partition_columns} in {base_path}: {missing_partition_instances}")

        return is_complete, missing_partition_instances

    def mark_complete(self,
                     base_path: str,
                     partition_columns: Optional[List[str]] = None,
                     partition_instances: Optional[List[List[str]]] = None) -> None:
        """
        标记表或分区完成（写入_COMPLETE标志文件）- URI-aware

        Args:
            base_path: 表的基础路径，可以包含URI scheme
            partition_columns: 分区列名列表（非分区表不传）
            partition_instances: 要标记的分区实例列表
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        complete_content = f"Completed at {timestamp}"

        # 处理空分区列（非分区表）
        if not partition_columns:
            complete_file = f"{base_path}/_COMPLETE"
            self.fileio.write_text(complete_file, complete_content)  # URI-aware!
            logger.info(f"Marked {base_path} as complete")
            return

        # 如果没有指定分区实例，不做任何操作
        if not partition_instances:
            logger.warning(f"No partition instances specified for marking complete")
            return

        # 标记每个分区实例
        for partition_instance in partition_instances:
            if len(partition_columns) != len(partition_instance):
                raise ValueError(
                    f"Number of partition columns ({len(partition_columns)}) "
                    f"and values ({len(partition_instance)}) don't match"
                )

            partition_path = build_partition_path(base_path, partition_columns, partition_instance)
            complete_file = f"{partition_path}/_COMPLETE"
            self.fileio.write_text(complete_file, complete_content)  # URI-aware!

        logger.info(f"Marked {len(partition_instances)} partition instances in {base_path} as complete")
