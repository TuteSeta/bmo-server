import subprocess
from typing import Optional


class DockerManager:
    """
    Utility for starting, stopping, and querying Docker containers.

    Intended for managing local services such as Ollama when they are
    run inside Docker rather than installed natively.

    Example usage:
        docker = DockerManager()
        docker.start_container(
            name="ollama",
            image="ollama/ollama",
            ports={"11434": "11434"},
            volumes={"/path/to/models": "/root/.ollama"},
        )
    """

    def start_container(
        self,
        name: str,
        image: str,
        ports: Optional[dict] = None,
        volumes: Optional[dict] = None,
    ) -> bool:
        """
        Run a Docker container in detached mode.

        Args:
            name:    Container name (--name).
            image:   Docker image to run.
            ports:   Mapping of {host_port: container_port} strings.
            volumes: Mapping of {host_path: container_path} strings.

        Returns:
            True if the container started successfully.
        """
        cmd = ["docker", "run", "-d", "--name", name]
        for host_port, container_port in (ports or {}).items():
            cmd += ["-p", f"{host_port}:{container_port}"]
        for host_path, container_path in (volumes or {}).items():
            cmd += ["-v", f"{host_path}:{container_path}"]
        cmd.append(image)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[DockerManager] Failed to start '{name}': {result.stderr.strip()}")
        return result.returncode == 0

    def stop_container(self, name: str) -> bool:
        """
        Force-remove a running or stopped container.

        Returns:
            True if the container was removed successfully.
        """
        result = subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[DockerManager] Failed to stop '{name}': {result.stderr.strip()}")
        return result.returncode == 0

    def is_running(self, name: str) -> bool:
        """Return True if a container with the given name is currently running."""
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name=^{name}$"],
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())

    def list_containers(self, all_containers: bool = False) -> list:
        """
        Return running containers as a list of dicts.

        Args:
            all_containers: If True, include stopped containers.

        Returns:
            List of dicts with 'name', 'status', and 'image' keys.
            Returns an empty list if Docker is unreachable.
        """
        cmd = ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}"]
        if all_containers:
            cmd.append("--all")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return []

        containers = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                containers.append({
                    "name": parts[0],
                    "status": parts[1],
                    "image": parts[2],
                })
        return containers
