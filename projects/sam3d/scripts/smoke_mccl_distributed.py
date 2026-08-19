#!/usr/bin/env python3
"""Minimal multi-node MCCL broadcast/all-reduce smoke test."""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.musa.set_device(local_rank)
    dist.init_process_group(backend="mccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"musa:{local_rank}")

        broadcast_value = torch.tensor([-1], dtype=torch.int64, device=device)
        if rank == 0:
            broadcast_value.fill_(12345)
        dist.broadcast(broadcast_value, src=0)

        reduced_value = torch.tensor([rank], dtype=torch.int64, device=device)
        dist.all_reduce(reduced_value)
        expected_sum = world_size * (world_size - 1) // 2
        if broadcast_value.item() != 12345 or reduced_value.item() != expected_sum:
            raise RuntimeError(
                f"collective mismatch on rank {rank}: "
                f"broadcast={broadcast_value.item()}, all_reduce={reduced_value.item()}, expected={expected_sum}"
            )
        dist.barrier()
        if rank == 0:
            print(
                f"MCCL_SMOKE_OK world_size={world_size} host={socket.gethostname()} "
                f"gid={os.environ.get('MCCL_IB_GID_INDEX')} "
                f"hca={os.environ.get('MCCL_IB_HCA')} "
                f"socket={os.environ.get('MCCL_SOCKET_IFNAME')}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
