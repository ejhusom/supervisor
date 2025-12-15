import os
import tomli
from pathlib import Path

class Config:
    """Simple configuration from config.toml with unified workspace."""

    def __init__(self, config_path: str):
        """Initialize config by loading the config.toml file."""
        self.config = self._load_config(config_path)

        # LLM settings
        self.config["provider"] = self.config.get("provider", "anthropic")
        self.config["model"] = self.config.get("model", "claude-sonnet-4-20250514")
        self.config["temperature"] = self.config.get("temperature", 0.0)
        self.config["max_tokens"] = self.config.get("max_tokens", 8192)

        # Agent loop settings
        self.config["max_iterations"] = self.config.get("max_iterations", 50)
        
        # Sandbox settings
        self.config["sandbox_timeout"] = self.config.get("sandbox_timeout", 100)

        # Tool execution settings
        self.config["tool_call_output_max_length"] = self.config.get("tool_call_output_max_length", 10000)
        self.config["truncate_tool_calls"] = self.config.get("truncate_tool_calls", True)

        # Logging settings
        self.config["log_dir"] = self.config.get("log_dir", "./logs")
        self.config["logging_enabled"] = self.config.get("logging_enabled", True)
        self.config["log_full_messages"] = self.config.get("log_full_messages", True)
        
        # Printer settings
        self.config["print_verbosity"] = self.config.get("print_verbosity", "normal")
        self.config["print_use_colors"] = self.config.get("print_use_colors", True)
        self.config["print_truncate_length"] = self.config.get("print_truncate_length", 200)
        
        # Unified workspace structure
        workspace_root = self.config.get("workspace", "./workspace")
        self.config["workspace"] = workspace_root
        self.config["workspace_data"] = f"{workspace_root}/data"
        self.config["workspace_agents"] = f"{workspace_root}/agents"
        self.config["workspace_tools"] = f"{workspace_root}/tools"

        # Preprocessing settings
        self.config["embeddings_chunk_size"] = self.config.get("embeddings_chunk_size", 100)
        
        # Create workspace directories
        for path in [
            self.config["workspace"],
            self.config["workspace_data"],
            self.config["workspace_agents"],
            self.config["workspace_tools"]
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

        # API key handling
        if "api_key" in self.config:
            self.config["api_key"] = self.config["api_key"]
        elif self.config["provider"] in ["anthropic", "openai"]:
            try:
                if self.config["provider"] == "anthropic":
                    self.config["api_key"] = os.environ["ANTHROPIC_API_KEY"]
                elif self.config["provider"] == "openai":
                    self.config["api_key"] = os.environ["OPENAI_API_KEY"]
            except KeyError:
                self.config["api_key"] = None
                print(f"API key not found for provider: {self.config['provider']}")
        else:
            self.config["api_key"] = None

        # Check workspace is writable
        try:
            test_file = Path(self.config["workspace"]) / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            print(f"Workspace not writable: {self.config['workspace']} - {e}")

    def _load_config(self, config_path: str) -> dict:
        """Load the configuration file."""
        try:
            with open(config_path, "rb") as file:
                return tomli.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        except tomli.TOMLDecodeError as e:
            raise ValueError(f"Error parsing config file: {e}")

config = Config("config.toml").config
