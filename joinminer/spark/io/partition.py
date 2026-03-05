"""
Partition utility functions for Spark table operations.

This module provides reusable utilities for working with partitioned tables,
including path construction and partition configuration parsing/validation.
"""

import itertools
import logging
from typing import Dict, List, Optional, Union
from collections import Counter

logger = logging.getLogger(__name__)


def parse_partition_spec(
    partition_columns: Optional[List[str]],
    partition_spec: Optional[Union[Dict[str, List[str]], List[List[str]], List[str]]],
    max_combinations: int = 100000
) -> Optional[List[List[str]]]:
    """
    解析并验证分区规范，转换为标准的分区实例列表。

    实现策略：先转换、后验证
    1. 将所有输入格式转换为二维数组（不做值验证）
    2. 统一验证二维数组（单一验证入口，确保一致性）

    支持三种输入格式：
    1. 字典格式: {'date': ['2021-01-01'], 'region': ['US', 'CN']}
       - 最明确，自动计算笛卡尔积
    2. 二维数组格式: [['2021-01-01', 'US'], ['2021-01-02', 'CN']]
       - 预计算的组合，直接使用（验证后）
    3. 一维数组格式: ['2021-01-01', '2021-01-02']
       - 单分区列的简写形式（仅当 len(partition_columns)==1 时有效）

    Args:
        partition_columns: 分区列名列表（例如 ['date', 'region']）
        partition_spec: 分区规范（用户输入），支持三种格式（见上）
        max_combinations: 最大允许组合数（默认100,000，超过会报错）

    Returns:
        标准化的分区实例列表 List[List[str]]，或 None（当两参数都为None时）

        单列示例: [['2021-01-01'], ['2021-01-02']]
        多列示例: [['2021-01-01', 'US'], ['2021-01-01', 'CN']]

    Raises:
        ValueError: 输入验证失败，包含详细错误信息

    Examples:
        >>> # 格式1：字典格式（自动笛卡尔积）
        >>> parse_partition_spec(['date'], {'date': ['2021-01-01', '2021-01-02']})
        [['2021-01-01'], ['2021-01-02']]

        >>> parse_partition_spec(
        ...     ['date', 'region'],
        ...     {'date': ['2021-01-01'], 'region': ['US', 'CN']}
        ... )
        [['2021-01-01', 'US'], ['2021-01-01', 'CN']]

        >>> # 格式2：二维数组（预计算组合）
        >>> parse_partition_spec(
        ...     ['date', 'region'],
        ...     [['2021-01-01', 'US'], ['2021-01-02', 'CN']]
        ... )
        [['2021-01-01', 'US'], ['2021-01-02', 'CN']]

        >>> # 格式3：一维数组（单分区列简写）
        >>> parse_partition_spec(['date'], ['2021-01-01', '2021-01-02'])
        [['2021-01-01'], ['2021-01-02']]

        >>> # 非分区表
        >>> parse_partition_spec(None, None)
        None
    """
    # 1. 处理完全为空的情况（非分区表）
    if partition_columns is None and partition_spec is None:
        return None

    # 2. 验证输入完整性
    if partition_columns is None or partition_spec is None:
        raise ValueError(
            "partition_columns and partition_spec must both be provided or both be None. "
            f"Got: partition_columns={partition_columns is not None}, "
            f"partition_spec={partition_spec is not None}"
        )

    if not partition_columns:
        raise ValueError("partition_columns cannot be empty list")

    # 3. 验证列名
    _validate_column_names(partition_columns)

    # 4. 根据输入类型转换为二维数组（不做值验证）
    if isinstance(partition_spec, dict):
        # 字典格式: 先验证键匹配，再转换
        _validate_keys_match(partition_columns, partition_spec)
        partition_instances_2d = _dict_to_2d(partition_columns, partition_spec)

    elif isinstance(partition_spec, list):
        if not partition_spec:
            raise ValueError("partition_spec cannot be empty list")

        # 检测是一维还是二维列表
        first_element = partition_spec[0]

        if isinstance(first_element, str):
            # 一维数组: 转换为二维
            partition_instances_2d = _1d_to_2d(partition_columns, partition_spec)

        elif isinstance(first_element, list):
            # 二维数组: 已经是目标格式，直接使用
            partition_instances_2d = partition_spec

        else:
            raise ValueError(
                f"Invalid partition_spec list element type: {type(first_element).__name__}. "
                f"Expected str (for 1D list) or list (for 2D list)."
            )

    else:
        raise ValueError(
            f"partition_spec must be dict, List[str], or List[List[str]]. "
            f"Got: {type(partition_spec).__name__}"
        )

    # 5. 统一验证二维数组
    _validate_partition_instances(partition_instances_2d, partition_columns, max_combinations)

    return partition_instances_2d


def _validate_column_names(partition_columns: List[str]) -> None:
    """验证分区列名有效性"""
    # 检查空字符串
    if any(not col or not col.strip() for col in partition_columns):
        raise ValueError(
            f"partition_columns contains empty or whitespace-only strings: {partition_columns}"
        )

    # 检查重复
    if len(partition_columns) != len(set(partition_columns)):
        duplicates = [col for col, count in Counter(partition_columns).items() if count > 1]
        raise ValueError(f"partition_columns contains duplicates: {duplicates}")

    # 检查特殊字符（会破坏分区路径）
    invalid_chars = set('=/*?\\')
    for col in partition_columns:
        if any(c in invalid_chars for c in col):
            raise ValueError(
                f"partition_columns['{col}'] contains invalid characters. "
                f"Cannot use: {invalid_chars}"
            )


def _validate_keys_match(partition_columns: List[str], partition_spec: Dict[str, List]) -> None:
    """验证字典键与列名匹配"""
    expected_keys = set(partition_columns)
    actual_keys = set(partition_spec.keys())

    if expected_keys != actual_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        raise ValueError(
            f"partition_columns and partition_spec keys don't match. "
            f"Expected: {sorted(expected_keys)}, Got: {sorted(actual_keys)}. "
            f"Missing: {sorted(missing) if missing else 'none'}, "
            f"Extra: {sorted(extra) if extra else 'none'}"
        )


def _validate_partition_value(value: str, path: str) -> None:
    """
    验证单个分区值的有效性。

    这是所有值验证的统一入口点，确保一致性。

    Args:
        value: 要验证的分区值
        path: 值在配置中的路径（用于错误消息）

    Raises:
        ValueError: 如果值无效
    """
    # 1. 类型检查
    if not isinstance(value, str):
        raise ValueError(
            f"{path} must be string, got {type(value).__name__}: {value}"
        )

    # 2. 特殊字符检查：路径分隔符
    if '/' in value or '\\' in value:
        raise ValueError(f"{path}='{value}' contains path separator ('/' or '\\\\')")

    # 3. 特殊字符检查：等号
    if '=' in value:
        raise ValueError(f"{path}='{value}' contains '=' character which breaks partition format")


def _validate_partition_instances(
    partition_instances: List[List[str]],
    partition_columns: List[str],
    max_combinations: int
) -> None:
    """
    统一验证分区实例列表。

    所有格式转换后都通过这个函数进行验证，确保验证逻辑一致。

    Args:
        partition_instances: 分区实例列表（二维数组）
        partition_columns: 分区列名列表
        max_combinations: 最大允许组合数

    Raises:
        ValueError: 如果验证失败
    """
    # 1. 检查非空
    if not partition_instances:
        raise ValueError("partition_spec cannot be empty")

    # 2. 检查数量限制
    if len(partition_instances) > max_combinations:
        raise ValueError(
            f"Too many partition instances: {len(partition_instances):,} exceeds limit {max_combinations:,}"
        )

    if len(partition_instances) > 10000:
        logger.warning(f"Large partition instance count: {len(partition_instances):,}")

    # 3. 验证每个分区实例
    num_columns = len(partition_columns)
    for i, partition_instance in enumerate(partition_instances):
        # 检查是列表
        if not isinstance(partition_instance, list):
            raise ValueError(
                f"partition_instance[{i}] must be list, got {type(partition_instance).__name__}"
            )

        # 检查列数
        if len(partition_instance) != num_columns:
            raise ValueError(
                f"partition_instance[{i}] has {len(partition_instance)} values, expected {num_columns}"
            )

        # 验证每个值
        for j, value in enumerate(partition_instance):
            _validate_partition_value(value, f"partition_instance[{i}][{j}]")


def _dict_to_2d(partition_columns: List[str], partition_dict: Dict[str, List[str]]) -> List[List[str]]:
    """
    纯转换：字典格式 → 二维数组格式（笛卡尔积）。

    Args:
        partition_columns: 分区列名列表
        partition_dict: 分区值字典

    Returns:
        二维分区数组
    """
    if len(partition_columns) == 1:
        # 单列：直接转换
        return [[val] for val in partition_dict[partition_columns[0]]]
    else:
        # 多列：笛卡尔积
        partition_values = [partition_dict[col] for col in partition_columns]
        return [list(combo) for combo in itertools.product(*partition_values)]


def _1d_to_2d(partition_columns: List[str], partition_1d: List[str]) -> List[List[str]]:
    """
    纯转换：一维数组 → 二维数组格式。

    Args:
        partition_columns: 分区列名列表（必须长度为1）
        partition_1d: 一维分区值列表

    Returns:
        二维分区数组

    Raises:
        ValueError: 如果分区列数不为1
    """
    if len(partition_columns) != 1:
        raise ValueError(
            f"1D format requires single column, got {len(partition_columns)}: {partition_columns}"
        )
    return [[val] for val in partition_1d]


def build_partition_path(base_path: str,
                        partition_columns: List[str],
                        partition_values: List[str]) -> str:
    """
    Build a Spark partition path while preserving the URI scheme.

    Constructs paths in the format: base_path/col1=val1/col2=val2/...

    Args:
        base_path: Base table path, may include URI scheme (e.g., 'hdfs:///data/table')
        partition_columns: List of partition column names (e.g., ['date', 'region'])
        partition_values: List of partition values (e.g., ['2024-01-01', 'US'])

    Returns:
        Complete partition path with URI scheme preserved

    Examples:
        >>> build_partition_path('hdfs:///data/table', ['date'], ['2024-01-01'])
        'hdfs:///data/table/date=2024-01-01'

        >>> build_partition_path('file:///local/db', ['date', 'region'],
        ...                      ['2024-01-01', 'US'])
        'file:///local/db/date=2024-01-01/region=US'

        >>> build_partition_path('/data/table', ['year', 'month'], ['2024', '01'])
        '/data/table/year=2024/month=01'
    """
    path_parts = [base_path.rstrip('/')]
    for col, val in zip(partition_columns, partition_values):
        path_parts.append(f"{col}={val}")
    return "/".join(path_parts)
