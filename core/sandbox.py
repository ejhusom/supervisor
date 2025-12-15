"""
Enhanced sandbox with Unix tool access.
Safe execution of both Python code and Unix commands.
"""

import subprocess
import tempfile
import json
import shlex
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import config

class Sandbox:
    """Execute Python code and Unix commands safely in subprocess."""
    
    # Allowed Unix commands (whitelist for security)
    ALLOWED_COMMANDS = {
        # File viewing
        'cat', 'echo', 'head', 'tail', 'less', 'more',
        # File operations
        'ls', 'find', 'wc', 'stat', 'file',
        # Text processing
        'grep', 'egrep', 'fgrep', 'sed', 'awk', 'cut', 'sort', 'uniq', 'tr',
        # Version control
        'git',
        # Compression
        'gzip', 'gunzip', 'tar', 'zip', 'unzip',
        # Other utilities
        'diff', 'comm', 'join', 'paste', 'tee', 'xargs', 'pwd',
    }
    
    def __init__(
        self,
        timeout: int = None,
        workspace: str = None,
    ):
        """
        Args:
            timeout: Execution timeout in seconds
            workspace: Directory for code execution (defaults to workspace/data)
        """
        self.timeout = timeout if timeout else config.get("sandbox_timeout")
        self.workspace = Path(workspace) if workspace else Path(config.get("workspace_data"))
        self.workspace.mkdir(parents=True, exist_ok=True)
    
    def execute(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute Python code in isolated process.
        
        Args:
            code: Python code to execute
            context: Optional variables to make available (as JSON)
        
        Returns:
            {
                "success": bool,
                "output": str,
                "error": str,
                "return_value": Any (if code uses `result = ...`)
            }
        """
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            dir=self.workspace,
            delete=False
        ) as f:
            # Wrap code to capture output
            wrapped_code = self._wrap_code(code, context)
            f.write(wrapped_code)
            f.flush()
            
            try:
                result = subprocess.run(
                    ['python', f.name],
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                    cwd=self.workspace
                )
                
                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr,
                    "return_value": self._extract_result(result.stdout)
                }
            
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Execution timeout ({self.timeout}s)",
                    "return_value": None
                }
            
            except Exception as e:
                return {
                    "success": False,
                    "output": "",
                    "error": str(e),
                    "return_value": None
                }
            
            finally:
                # Cleanup
                Path(f.name).unlink(missing_ok=True)
    
    def execute_command(
        self,
        command: str,
        args: List[str] = None,
        input_data: str = None,
        cwd: str = None
    ) -> Dict[str, Any]:
        """
        Execute a Unix command safely.
        
        Args:
            command: Command name (e.g., 'grep', 'cat')
            args: Command arguments as list
            input_data: Optional stdin data
            cwd: Working directory (defaults to workspace, must be within workspace)
        
        Returns:
            {
                "success": bool,
                "output": str,
                "error": str,
                "returncode": int
            }
        """
        # Security check: command must be whitelisted
        if command not in self.ALLOWED_COMMANDS:
            return {
                "success": False,
                "output": "",
                "error": f"Command '{command}' not allowed. Allowed: {sorted(self.ALLOWED_COMMANDS)}",
                "returncode": 1
            }
        
        # Set working directory
        if cwd:
            cwd_path = Path(cwd).resolve()
            # Security: must be within workspace
            try:
                cwd_path.relative_to(self.workspace.resolve())
            except ValueError:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Working directory must be within workspace: {self.workspace}",
                    "returncode": 1
                }
        else:
            cwd_path = self.workspace
        
        # Build command
        cmd = [command]
        if args:
            cmd.extend(args)
        
        try:
            result = subprocess.run(
                cmd,
                input=input_data,
                timeout=self.timeout,
                capture_output=True,
                text=True,
                cwd=cwd_path
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "returncode": result.returncode
            }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Command timeout ({self.timeout}s)",
                "returncode": 124
            }
        
        except FileNotFoundError:
            return {
                "success": False,
                "output": "",
                "error": f"Command '{command}' not found on system",
                "returncode": 127
            }
        
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "returncode": 1
            }
    
    def execute_shell(
        self,
        command_line: str,
        cwd: str = None
    ) -> Dict[str, Any]:
        """
        Execute a shell command line (allows pipes, redirects).
        
        SECURITY WARNING: This is less safe than execute_command().
        Only use with trusted input or after validation.
        
        Args:
            command_line: Shell command line (e.g., "grep ERROR log.txt | wc -l")
            cwd: Working directory (must be within workspace)
        
        Returns:
            Same format as execute_command
        """
        # Parse command to check first command is allowed
        try:
            first_cmd = shlex.split(command_line)[0]
            if first_cmd not in self.ALLOWED_COMMANDS:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Command '{first_cmd}' not allowed",
                    "returncode": 1
                }
        except (IndexError, ValueError):
            return {
                "success": False,
                "output": "",
                "error": "Invalid command line",
                "returncode": 1
            }
        
        # Set working directory
        if cwd:
            cwd_path = Path(cwd).resolve()
            try:
                cwd_path.relative_to(self.workspace.resolve())
            except ValueError:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Working directory must be within workspace",
                    "returncode": 1
                }
        else:
            cwd_path = self.workspace
        
        try:
            result = subprocess.run(
                command_line,
                shell=True,
                timeout=self.timeout,
                capture_output=True,
                text=True,
                cwd=cwd_path
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "returncode": result.returncode
            }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Command timeout ({self.timeout}s)",
                "returncode": 124
            }
        
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "returncode": 1
            }

    def _wrap_code(self, code: str, context: Dict[str, Any] = None) -> str:
        """Wrap code to capture result."""
        wrapper = []
        
        # Add context if provided - use unlikely variable name to avoid collision
        if context:
            wrapper.append("import json as __iexplain_json__")
            wrapper.append(f"__iexplain_context__ = __iexplain_json__.loads('{json.dumps(context)}')")
            wrapper.append("")
        
        # Add user code
        wrapper.append(code)
        wrapper.append("")
        
        # Capture result if assigned
        wrapper.append("""
# Try to capture result
if 'result' in locals():
    print('__RESULT__:', result)
""")

        return "\n".join(wrapper)
    
    def _extract_result(self, output: str) -> Any:
        """Extract result from output."""
        for line in output.split('\n'):
            if line.startswith('__RESULT__:'):
                result_str = line.split('__RESULT__:', 1)[1].strip()
                try:
                    return json.loads(result_str)
                except:
                    return result_str
        return None
