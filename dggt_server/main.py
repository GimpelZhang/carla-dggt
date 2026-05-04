# SPDX-FileCopyrightText: © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
DGGT gRPC Server Entry Point

Main entry point for starting the DGGT gRPC service.
"""

import argparse
import logging
import sys
from concurrent import futures

import grpc

from .config import DGGTServerConfig
from .scene_manager import DGGTSceneManager


logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        stream=sys.stdout
    )


def serve(
    host: str,
    port: int,
    scene_path: str,
    config_path: str = None,
    max_workers: int = 10,
    device: str = "cuda",
    log_level: str = "INFO"
) -> None:
    """
    Start the DGGT gRPC server

    Args:
        host: Server host address
        port: Server port
        scene_path: Path to scene directory (required)
        config_path: Path to YAML config file (optional)
        max_workers: Maximum thread pool workers
        device: Device for rendering (cuda/cpu)
        log_level: Logging level
    """
    # Setup logging
    setup_logging(log_level)

    # Load configuration
    if config_path:
        logger.info(f"Loading configuration from: {config_path}")
        config = DGGTServerConfig.from_yaml(config_path)
        # Override with CLI arguments if provided
        if host != "0.0.0.0":
            config.grpc_host = host
        if port != 50052:
            config.grpc_port = port
        if scene_path:
            config.scene_base_path = scene_path
        if max_workers != 10:
            config.max_workers = max_workers
        if device != "cuda":
            config.device = device
    else:
        # Use CLI arguments directly
        config = DGGTServerConfig(
            grpc_host=host,
            grpc_port=port,
            max_workers=max_workers,
            device=device,
            scene_base_path=scene_path
        )

    logger.info(f"Starting DGGT gRPC server on {config.grpc_host}:{config.grpc_port}")
    logger.info(f"Scene path: {config.scene_base_path}")
    logger.info(f"Max workers: {config.max_workers}")
    logger.info(f"Device: {config.device}")

    # Initialize scene manager
    scene_manager = DGGTSceneManager(
        scene_base_path=config.scene_base_path,
        default_fps=config.default_fps
    )

    try:
        scene_manager.initialize(auto_discover=config.auto_discover)
        logger.info(f"Scene manager initialized with {len(scene_manager.list_scenes())} scenes")
    except Exception as e:
        logger.error(f"Failed to initialize scene manager: {e}")
        sys.exit(1)

    # Register scene ID mappings from config
    for external_id, internal_id in config.scene_id_mappings.items():
        scene_manager.register_scene(external_id, internal_id)

    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))

    # Add DGGTService servicer
    from .dggt_service import DGGTService
    from nre.grpc.protos import sensorsim_pb2_grpc

    dggt_service = DGGTService(
        scene_manager=scene_manager,
        config=config
    )
    sensorsim_pb2_grpc.add_SensorsimServiceServicer_to_server(dggt_service, server)

    server.add_insecure_port(f"{config.grpc_host}:{config.grpc_port}")

    # Start server
    server.start()
    logger.info(f"DGGT gRPC server started and listening on {config.grpc_host}:{config.grpc_port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down...")
        server.stop(grace=5)
        logger.info("Server stopped gracefully")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="DGGT gRPC Server - Dynamic Gaussian Splatting Scene Service"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=50052,
        help="Server port (default: 50052)"
    )

    parser.add_argument(
        "--scene-path",
        type=str,
        required=True,
        help="Path to scene directory (required)"
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to YAML configuration file (optional)"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum thread pool workers (default: 10)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for rendering (default: cuda)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    serve(
        host=args.host,
        port=args.port,
        scene_path=args.scene_path,
        config_path=args.config_path,
        max_workers=args.max_workers,
        device=args.device,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()