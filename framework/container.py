"""
framework/container.py
~~~~~~~~~~~~~~~~~~~~~~
Docker SDK wrapper for level container lifecycle management.

Each level runs inside its own isolated Docker container.
This module handles: start, stop, exec, snapshot, and cleanup.
"""

from __future__ import annotations

import io
import logging
import os
import tarfile
import time
from contextlib import contextmanager
from typing import Any, Generator

import docker
from docker.errors import DockerException, ImageNotFound, NotFound
from docker.models.containers import Container

logger = logging.getLogger(__name__)

# How long to wait (s) for a container to reach "running" state.
_START_TIMEOUT = 30


class ContainerError(Exception):
    """Raised when a container operation fails in a non-recoverable way."""


class LevelContainer:
    """
    Manages the Docker container lifecycle for a single benchmark level.

    Parameters
    ----------
    level_cfg : dict
        Parsed ``container`` block from a level YAML file.
    framework_cfg : dict
        Framework-wide defaults (from config.yaml → container section).
    level_id : str
        Human-readable level identifier used for container naming.
    """

    def __init__(
        self,
        level_cfg: dict[str, Any],
        framework_cfg: dict[str, Any],
        level_id: str,
    ) -> None:
        self.level_id = level_id
        self._cfg = level_cfg
        self._fw = framework_cfg

        self._container: Container | None = None
        self._client: docker.DockerClient | None = None  # lazy — initialised in start()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Connect to Docker daemon, pull image (if needed), and start the container."""
        # Lazy Docker client init — gives a clear error only when actually needed
        self._client = self._make_client()
        image = self._cfg["image"]
        pull_policy = self._fw.get("pull_policy", "if_not_present")
        self._ensure_image(image, pull_policy)

        container_name = f"benchb0t-{self.level_id}-{int(time.time())}"
        env = self._resolve_env(self._cfg.get("env", {}))

        # Publish preview port if the level declares one.
        # preview_cfg comes from the top-level "preview:" key in the level YAML,
        # which is passed into level_cfg by the runner.
        preview_port = self._cfg.get("preview_port")
        ports: dict[str, int] | None = None
        if preview_port:
            port_int = int(preview_port)
            # Map host port → container port (same number for simplicity).
            # docker-py format: {"container_port/tcp": host_port}
            ports = {f"{port_int}/tcp": port_int}
            logger.info("Publishing port %d → host:%d for level preview", port_int, port_int)

        # IMPORTANT: do NOT pass network_mode when publishing ports.
        # Passing network_mode="bridge" alongside ports= silently drops the
        # port binding on Docker Desktop (macOS/Windows). Docker's default
        # bridge network (no explicit --network flag) supports port publishing
        # AND has internet access, so it's correct for both cases.
        network_mode = None if ports else self._fw.get("network_mode", "bridge")

        logger.info("Starting container %s (image=%s)", container_name, image)
        try:
            run_kwargs: dict = dict(
                image=image,
                name=container_name,
                # Keep the container alive indefinitely — the agent drives it via exec()
                command=["sleep", "infinity"],
                detach=True,
                remove=False,                  # we clean up manually after scoring
                working_dir=self._cfg.get("working_dir", "/workspace"),
                environment=env,
                volumes=self._parse_volumes(self._cfg.get("volumes", [])),
                cpu_quota=self._cpu_quota(),
                mem_limit=self._cfg.get(
                    "memory_limit", self._fw.get("default_memory_limit", "4g")
                ),
            )
            if network_mode:
                run_kwargs["network_mode"] = network_mode
            if ports:
                run_kwargs["ports"] = ports

            self._container = self._client.containers.run(**run_kwargs)
        except DockerException as exc:
            raise ContainerError(f"Failed to start container: {exc}") from exc

        self._wait_for_running()

        # Log the actual port bindings Docker assigned (for diagnostics)
        if ports:
            try:
                self._container.reload()
                actual = self._container.ports
                logger.info("Actual port bindings: %s", actual)
            except Exception:
                pass

        # Install declared packages before the setup script runs
        self._install_packages(self._cfg.get("packages", {}))

        # Run optional setup script inside the container.
        # Pass the script directly — self.exec() already wraps in ["sh", "-c", …],
        # so the script string is handed to sh as a single argument.
        # This correctly handles multi-line scripts and embedded single quotes,
        # unlike wrapping in f"sh -c '{script}'" which breaks on any ' in the script.
        setup_script = self._cfg.get("setup_script")
        if setup_script:
            exit_code, out = self.exec(setup_script)
            if exit_code != 0:
                raise ContainerError(
                    f"Setup script failed (exit {exit_code}):\n{out[:600]}"
                )

        logger.info("Container %s is ready", container_name)

    def exec(self, command: str, *, workdir: str | None = None) -> tuple[int, str]:
        """
        Run a shell command inside the container.

        Returns
        -------
        (exit_code, output)
            exit_code 0 means success.
        """
        self._assert_running()
        wd = workdir or self._cfg.get("working_dir", "/workspace")
        logger.debug("exec [%s] $ %s", self.level_id, command)

        exit_code, output = self._container.exec_run(  # type: ignore[union-attr]
            cmd=["sh", "-c", command],
            workdir=wd,
            demux=False,
        )
        decoded = output.decode("utf-8", errors="replace") if output else ""
        if exit_code != 0:
            logger.warning("exec exited %d: %s", exit_code, decoded[:200])
        return exit_code, decoded

    def read_file(self, path: str) -> str:
        """Read a file from inside the container and return its content."""
        exit_code, content = self.exec(f"cat {path}")
        if exit_code != 0:
            raise ContainerError(f"Cannot read {path}: {content}")
        return content

    def write_file(self, path: str, content: str) -> None:
        """
        Write content to a file inside the container.

        Uses Docker's put_archive API to stream the file bytes directly —
        this completely avoids shell quoting and works for any content
        including JSX, Python, shell scripts, or binary data.
        """
        self._assert_running()
        encoded = content.encode("utf-8")
        filename = os.path.basename(path)
        dest_dir = os.path.dirname(path) or "/"

        # Ensure the parent directory exists first
        self.exec(f"mkdir -p {dest_dir}")

        # Build a minimal tar archive in memory containing just this one file
        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode="w") as tar:
            info = tarfile.TarInfo(name=filename)
            info.size = len(encoded)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(encoded))
        tarstream.seek(0)

        try:
            ok = self._container.put_archive(dest_dir, tarstream)  # type: ignore[union-attr]
        except Exception as exc:
            raise ContainerError(f"Cannot write {path}: {exc}") from exc
        if not ok:
            raise ContainerError(f"put_archive returned False for {path}")

    def snapshot(self) -> str:
        """
        Commit the container state as a new image and return the image tag.
        Useful for saving mid-level snapshots.
        """
        self._assert_running()
        tag = f"benchb0t-snapshot-{self.level_id}-{int(time.time())}"
        logger.info("Snapshotting container → %s", tag)
        self._container.commit(repository="benchb0t", tag=tag)  # type: ignore[union-attr]
        return f"benchb0t:{tag}"

    def stop(self, *, remove: bool = True) -> None:
        """Stop (and optionally remove) the container."""
        if self._container is None:
            return
        try:
            logger.info("Stopping container %s", self._container.name)
            self._container.stop(timeout=10)
            if remove:
                self._container.remove(force=True)
                logger.debug("Container %s removed", self._container.name)
        except NotFound:
            pass  # already gone
        except DockerException as exc:
            logger.warning("Error stopping container: %s", exc)
        finally:
            self._container = None

    # ── Context manager ───────────────────────────────────────────────────────

    @contextmanager
    def session(self) -> Generator["LevelContainer", None, None]:
        """
        Context manager that starts the container on entry and always stops it
        on exit — even if the level raises an exception.

        Usage::

            with LevelContainer(cfg, fw_cfg, "l1").session() as c:
                code, out = c.exec("ls /workspace")
        """
        try:
            self.start()
            yield self
        finally:
            self.stop()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _install_packages(self, packages: dict[str, list[str]]) -> None:
        """
        Install packages declared in the level YAML ``container.packages`` block.

        Supported package managers:
          apt  — system packages  (apt-get install -y)
          pip  — Python packages  (pip install --quiet)
          npm  — Node packages    (npm install -g)
          gem  — Ruby gems        (gem install)

        Example YAML::

          container:
            packages:
              apt: [curl, git, jq]
              pip: [requests, pandas]
              npm: [lodash]
        """
        if not packages:
            return

        # apt
        apt_pkgs = packages.get("apt", [])
        if apt_pkgs:
            pkg_list = " ".join(apt_pkgs)
            logger.info("Installing apt packages: %s", pkg_list)
            exit_code, out = self.exec(
                f"apt-get update -qq && apt-get install -y --no-install-recommends {pkg_list}"
            )
            if exit_code != 0:
                logger.warning("apt install failed (exit %d): %s", exit_code, out[:300])

        # pip
        pip_pkgs = packages.get("pip", [])
        if pip_pkgs:
            pkg_list = " ".join(pip_pkgs)
            logger.info("Installing pip packages: %s", pkg_list)
            exit_code, out = self.exec(f"pip install --quiet {pkg_list}")
            if exit_code != 0:
                logger.warning("pip install failed (exit %d): %s", exit_code, out[:300])

        # npm
        npm_pkgs = packages.get("npm", [])
        if npm_pkgs:
            pkg_list = " ".join(npm_pkgs)
            logger.info("Installing npm packages: %s", pkg_list)
            exit_code, out = self.exec(f"npm install -g {pkg_list}")
            if exit_code != 0:
                logger.warning("npm install failed (exit %d): %s", exit_code, out[:300])

        # gem
        gem_pkgs = packages.get("gem", [])
        if gem_pkgs:
            pkg_list = " ".join(gem_pkgs)
            logger.info("Installing gem packages: %s", pkg_list)
            exit_code, out = self.exec(f"gem install {pkg_list}")
            if exit_code != 0:
                logger.warning("gem install failed (exit %d): %s", exit_code, out[:300])

    @staticmethod
    def _make_client() -> docker.DockerClient:
        host = os.getenv("DOCKER_HOST")
        try:
            client = docker.from_env() if not host else docker.DockerClient(base_url=host)
            client.ping()
            return client
        except DockerException as exc:
            raise ContainerError(
                f"Cannot connect to Docker daemon. Is Docker running? ({exc})"
            ) from exc

    def _ensure_image(self, image: str, pull_policy: str) -> None:
        if pull_policy == "never":
            return
        if pull_policy == "always":
            logger.info("Pulling image %s (policy=always)", image)
            self._client.images.pull(image)
            return
        # if_not_present
        try:
            self._client.images.get(image)
            logger.debug("Image %s already present, skipping pull", image)
        except ImageNotFound:
            logger.info("Image %s not found locally — pulling", image)
            self._client.images.pull(image)

    def _wait_for_running(self) -> None:
        deadline = time.time() + _START_TIMEOUT
        while time.time() < deadline:
            self._container.reload()  # type: ignore[union-attr]
            status = self._container.status  # type: ignore[union-attr]
            if status == "running":
                return
            if status in ("exited", "dead"):
                logs = self._container.logs().decode("utf-8", errors="replace")  # type: ignore[union-attr]
                raise ContainerError(
                    f"Container exited immediately (status={status}). Logs:\n{logs}"
                )
            time.sleep(0.5)
        raise ContainerError(f"Container did not reach 'running' within {_START_TIMEOUT}s")

    def _assert_running(self) -> None:
        if self._container is None:
            raise ContainerError("Container is not started. Call .start() first.")

    def _cpu_quota(self) -> int:
        """Convert CPU limit string (e.g. '2') to Docker cpu_quota microseconds."""
        raw = self._cfg.get(
            "cpu_limit", self._fw.get("default_cpu_limit", "2")
        )
        return int(float(raw) * 100_000)  # 100_000 µs = 1 CPU period

    @staticmethod
    def _resolve_env(env: dict[str, str]) -> dict[str, str]:
        """
        Substitute ${ENV_VAR} references in environment values from the host.
        """
        resolved = {}
        for key, value in env.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                resolved[key] = os.getenv(var_name, "")
            else:
                resolved[key] = str(value)
        return resolved

    @staticmethod
    def _parse_volumes(volumes: list[str]) -> dict[str, dict[str, str]]:
        """
        Parse volume strings like ``"./data:/workspace/data:ro"`` into the
        format expected by the Docker SDK.
        """
        parsed: dict[str, dict[str, str]] = {}
        for vol in volumes:
            parts = vol.split(":")
            host_path = os.path.abspath(parts[0])
            container_path = parts[1] if len(parts) > 1 else parts[0]
            mode = parts[2] if len(parts) > 2 else "rw"
            parsed[host_path] = {"bind": container_path, "mode": mode}
        return parsed
