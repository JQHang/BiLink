"""Distributed training setup utilities."""

import os
import random
import socket
import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def find_free_port(start: int = 29000, end: int = 29100) -> int:
    """Find an available port for distributed training.

    Args:
        start: Start of port range.
        end: End of port range.

    Returns:
        An available port number.
    """
    while True:
        try:
            sock = socket.socket()
            port = random.randint(start, end)
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            continue


def setup_ddp(
    rank: int,
    world_size: int,
    port: int,
    device_type: str,
    master_addr: str = "localhost",
) -> None:
    """Initialize distributed process group.

    Supports both CUDA (NCCL) and NPU (HCCL) backends.

    Args:
        rank: Current process rank.
        world_size: Total number of processes.
        port: Port for communication.
        device_type: "cuda" or "npu".
        master_addr: Master address (default "localhost").
    """
    # 1. Set env vars if not already set
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = master_addr
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(port)

    logger.info(f"Rank {rank}: MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")

    # 2. Initialize process group (always the same way)
    if not dist.is_initialized():
        backend = "nccl" if device_type == "cuda" else "hccl"
        dist.init_process_group(backend, rank=rank, world_size=world_size)


def get_device(rank: int, device_type: str) -> str:
    """Get device string for given rank.

    Args:
        rank: Process rank.
        device_type: "cuda", "npu", or "cpu".

    Returns:
        Device string like "cuda:0", "npu:1", or "cpu".
    """
    if device_type == "cpu":
        return "cpu"
    return f"{device_type}:{rank}"
