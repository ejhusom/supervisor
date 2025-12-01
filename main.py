"""
iExplain: Self-Modifying Agentic System

Entry point for the supervisor-driven architecture.
"""
import argparse
import os
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import config
from core.llm_client import LLMClient
from core.preprocessor import Preprocessor, SQLiteLogIngestion, EmbeddingRAG
from core.supervisor import Supervisor
from core.ui import get_ui
from registry.tool_registry import ToolRegistry
from registry.agent_registry import AgentRegistry
from workflows import SimpleWorkflow, EvaluatorWorkflow, MultiStageWorkflow, PredefinedMultiStageWorkflow

def setup_preprocessing(log_path: Path, use_embeddings: bool = False, chunk_size: int = 0) -> Preprocessor:
    """
    Setup and run preprocessing on log files.
    
    Args:
        log_path: Path to log file or directory
        use_embeddings: Whether to create embeddings for RAG
        chunk_size: Size of chunks for embeddings
    
    Returns:
        Configured and executed Preprocessor
    """
    
    preprocessor = Preprocessor()
    
    # Always add SQLite ingestion
    print("  Adding preprocessing step: SQLite Log Ingestion")
    preprocessor.add_step(SQLiteLogIngestion())
    
    # Optionally add embeddings
    if use_embeddings:
        if chunk_size <= 0:
            chunk_size = config.get("embeddings_chunk_size", 100)

        print(f"  Adding preprocessing step: Embedding RAG (chunk_size={chunk_size})")
        try:
            preprocessor.add_step(EmbeddingRAG(chunk_size=chunk_size))
        except ImportError as e:
            print(f"  Warning: Could not initialize embeddings: {e}")
            print("  Install sentence-transformers: pip install sentence-transformers")
            print("  Continuing without embeddings...")
    
    print(f"\n  Processing logs from: {log_path}\n")
    
    # Run preprocessing
    metadata = preprocessor.process(log_path)
    
    print("\n  Preprocessing complete!")
    
    return preprocessor

def main():
    """Run the supervisor agent."""
    ui = get_ui()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="iExplain: Agentic system for log analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode without preprocessing
  python main.py

  # Interactive mode with preprocessing
  python main.py --preprocess workspace/data/logs.log

  # Single task with preprocessing
  python main.py --preprocess workspace/data/*.log "Find all ERROR logs"

  # With embeddings for semantic search
  python main.py --preprocess workspace/data/logs.log --embeddings
        """
    )
    
    parser.add_argument(
        '--preprocess', '-p',
        type=str,
        metavar='LOG_PATH',
        help='Path to log file(s) to preprocess (enables preprocessing mode)'
    )
    
    parser.add_argument(
        '--embeddings', '-e',
        action='store_true',
        help='Create embeddings for semantic search (requires sentence-transformers)'
    )

    parser.add_argument(
        '--workflow', '-w',
        type=str,
        default='simple',
        choices=['simple', 'evaluator', 'multi_stage', 'analysis', 'research', 'transformation'],
        help='Workflow type: simple (default), evaluator (retry with evaluation), '
             'multi_stage (custom stages), or predefined: analysis, research, transformation'
    )
    
    parser.add_argument(
        '--stages',
        type=str,
        nargs='*',
        help='Stage descriptions for multi_stage workflow (e.g., --stages "Parse logs" "Analyze" "Report")'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=3,
        help='Max iterations for evaluator workflow (default: 3)'
    )
    
    parser.add_argument(
        'task',
        nargs='*',
        help='Task to execute (if omitted, starts interactive mode)'
    )
    
    args = parser.parse_args()
    
    ui = get_ui()
    ui.header("iExplain: Self-Modifying Agentic System")

    preprocessor = None
    if args.preprocess:
        log_path = Path(args.preprocess)

        if not log_path.exists():
            print(f"Log file not found: {log_path}")
            sys.exit(1)

        ui.section("Preprocessing Setup")

        preprocessor = setup_preprocessing(
            log_path=log_path,
            use_embeddings=args.embeddings,
            chunk_size=config.get("embeddings_chunk_size", 100)
        )

    # Initialize components
    ui.section("Initializing system...")
    
    llm_client = LLMClient(
        provider=config.get("provider", None),
        api_key=config.get("api_key", None),
        model=config.get("model", None)
    )
    tool_registry = ToolRegistry()
    agent_registry = AgentRegistry()
    
    supervisor = Supervisor(
        llm_client=llm_client,
        tool_registry=tool_registry,
        agent_registry=agent_registry,
        instructions_dir="instructions",
        preprocessor=preprocessor
    )
    
    ui.info(f"LLM client initialized (provider: {llm_client.provider}, model: {llm_client.model})")
    ui.info(f"Workspace: {config.get('workspace')}")
    ui.detail("Data", config.get('workspace_data'))
    ui.detail("Tools", config.get('workspace_tools'))
    ui.detail("Agents", config.get('workspace_agents'))
    ui.info(f"Tool registry initialized ({len(tool_registry.list_tools())} tools)")
    ui.info(f"Agent registry initialized ({len(agent_registry.list_agents())} agents)")

    if preprocessor:
        preprocessing_tools = preprocessor.get_all_tools()
        ui.info(f"Preprocessor enabled ({len(preprocessing_tools)} tools added)")
        ui.detail("Tools", ', '.join(preprocessing_tools.keys()))

    # Initialize workflow
    workflow_type = args.workflow
    
    if workflow_type == 'simple':
        workflow = SimpleWorkflow(supervisor)
    
    elif workflow_type == 'evaluator':
        workflow = EvaluatorWorkflow(
            supervisor,
            max_iterations=args.max_iterations,
            verbose=True
        )
        ui.workflow_info("evaluator", f"Evaluator workflow initialized (max_iterations={args.max_iterations})")
    
    elif workflow_type == 'multi_stage':
        if not args.stages:
            print("Error: --stages required for multi_stage workflow")
            print("Example: --workflow multi_stage --stages 'Parse data' 'Analyze' 'Summarize'")
            sys.exit(1)
        workflow = MultiStageWorkflow(
            supervisor,
            stages=args.stages,
            verbose=True
        )
        ui.workflow_info("multi_stage", f"Multi-stage workflow initialized ({len(args.stages)} stages)")
    
    elif workflow_type == 'analysis':
        workflow = PredefinedMultiStageWorkflow.analysis_workflow(supervisor)
        ui.workflow_info("analysis", "Analysis workflow initialized (parse → analyze → summarize)")
    
    elif workflow_type == 'research':
        workflow = PredefinedMultiStageWorkflow.research_workflow(supervisor)
        ui.workflow_info("research", "Research workflow initialized (gather → synthesize → conclude)")
    
    elif workflow_type == 'transformation':
        workflow = PredefinedMultiStageWorkflow.transformation_workflow(supervisor)
        ui.workflow_info("transformation", "Transformation workflow initialized (extract → transform → format)")
    
    else:
        # Fallback to simple
        workflow = SimpleWorkflow(supervisor)
    
    ui.info("Supervisor ready")
    ui.header_end()

    # Interactive mode or single task
    task_text = " ".join(args.task).strip() if args.task else None

    if task_text:
        # Single task from command line
        ui.task(task_text)
        ui.separator()
        print()
        
        result = workflow.run(task_text)
        
        ui.result_header()
        print(result["content"])
        ui.result_end()
        
        if result["tool_calls"]:
            print(f"Tools used: {len(result['tool_calls'])}")
            for tc in result["tool_calls"]:
                print(f"  └─ {tc['name']}")
            print()
    
    else:
        # Interactive REPL
        print("Interactive mode. Type '/exit' to quit.")
        print("Type '/help' for available commands.")
        print()
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['/exit', '/quit', '/bye']:
                    ui.success("Goodbye!")
                    break
                
                if user_input.lower() == '/help':
                    print_help(preprocessor is not None)
                    continue
                
                if user_input.lower() == '/tools':
                    tools_info = supervisor._list_tools()
                    print(f"Registry tools: {tools_info['registry_tools']}")
                    print(f"Standard tools: {tools_info['standard_tools']}")
                    if preprocessor:
                        print(f"Preprocessing tools: {tools_info['preprocessing_tools']}")
                    continue
                
                if user_input.lower() == '/agents':
                    print(f"Created agents: {agent_registry.list_agents()}")
                    continue
                
                # Execute task
                print()
                result = workflow.run(user_input)
                
                print()
                ui.separator()
                print(result["content"])
                ui.separator()
                print()
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type '/exit' to quit.")
            
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

def print_help(preprocessing_enabled: bool):
    """Print help message."""
    print("""
Commands:
  <task>      Execute a task with the supervisor
  /tools      List available tools
  /agents     List created agents
  /help       Show this help
  /exit       Quit
""")
    
    if preprocessing_enabled:
        print("""  /stats      Show preprocessing statistics

Preprocessing tools available:
  - query_logs_from_sqlite_database(sql, params): Execute SQL queries on logs
  - get_error_logs_from_sqlite_database(limit): Get ERROR-level logs
  - search_logs_from_sqlite_database(keyword, limit): Keyword search
  - get_log_stats_from_sqlite_database(): Get statistics overview
  - search_similar_from_embeddings(query, top_k): Semantic search (if embeddings enabled)
""")
    else:
        print("""
Standard tools available to all agents:
  - execute_python: Run Python code in sandbox
  - run_command: Execute Unix commands (grep, awk, etc.)
  - run_shell: Execute shell command lines with pipes
  - read_file: Read files from workspace
  - write_file: Write files to workspace
  - list_files: List directory contents
  - pwd: Show current working directory
""")
    
    print("""
Examples:
  >>> Create a tool to parse OpenStack logs
  >>> Analyze error.log for anomalies
  >>> Create an agent specialized in log analysis
""")
    
    if preprocessing_enabled:
        print("""
Preprocessing examples:
  >>> How many ERROR logs are there?
  >>> Find all connection timeout errors
  >>> What are the most common error patterns?
  >>> Search for logs containing "database"
  >>> Show me logs from the nova.compute component
""")
    
    print("\nThe supervisor will create tools and agents as needed.")

if __name__ == "__main__":
    main()