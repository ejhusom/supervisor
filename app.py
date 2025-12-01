"""
Simple Streamlit UI for iExplain.

Run with: streamlit run app.py
"""

import streamlit as st
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import config
from core.llm_client import LLMClient
from core.supervisor import Supervisor
from registry.tool_registry import ToolRegistry
from registry.agent_registry import AgentRegistry

# Page config
st.set_page_config(
    page_title="iExplain",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'supervisor' not in st.session_state:
    st.session_state.supervisor = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'logs_dir' not in st.session_state:
    st.session_state.logs_dir = Path(config.get("log_dir"))

def init_supervisor(provider, model, api_key, temperature, max_tokens):
    """Initialize the supervisor with given config."""
    llm_client = LLMClient(
        provider=provider,
        api_key=api_key,
        model=model
    )
    tool_registry = ToolRegistry()
    agent_registry = AgentRegistry()
    
    supervisor = Supervisor(
        llm_client=llm_client,
        tool_registry=tool_registry,
        agent_registry=agent_registry,
        instructions_dir="instructions"
    )
    
    # Update agent config
    supervisor.agent.temperature = temperature
    supervisor.agent.max_tokens = max_tokens
    
    return supervisor

def get_log_files():
    """Get all log files sorted by date."""
    logs_dir = st.session_state.logs_dir
    if not logs_dir.exists():
        return []
    
    log_files = list(logs_dir.glob("*.json"))
    log_files.sort(reverse=True)  # Newest first
    return log_files

def load_log(log_file):
    """Load a log file."""
    with open(log_file, 'r') as f:
        return json.load(f)

def display_log(log):
    """Display log in a nice format."""
    # Header
    st.markdown(f"### Session: {log['session_id']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Agents", len(log.get('interactions', [])))
    with col2:
        total_iters = sum(len(i['iterations']) for i in log.get('interactions', []))
        st.metric("Iterations", total_iters)
    with col3:
        try:
            st.metric("Duration", f"{log.get('duration', 0):.2f}s")
        except (TypeError, ValueError):
            st.metric("Duration", "N/A")
    
    # Task
    st.markdown("**Task:**")
    st.info(log['task'])
    
    # Config
    with st.expander("Configuration"):
        st.json(log['config'])
    
    # Final result
    st.markdown("**Final Result:**")
    st.success(log.get('final_result', 'No result'))
    
    # Interactions
    st.markdown("**Interactions:**")
    
    for interaction in log.get('interactions', []):
        agent_name = interaction['agent']
        parent = interaction.get('parent')
        duration = interaction.get('duration', 0)
        
        try:
            header = f"**{agent_name}** ({duration:.2f}s)"
        except (TypeError, ValueError):
            header = f"**{agent_name}** (N/A)"

        if parent:
            header += f" _← called by {parent}_"
        
        with st.expander(header, expanded=False):
            st.markdown(f"**Message:** {interaction['message']}")
            
            # Iterations
            for iter_data in interaction.get('iterations', []):
                iter_num = iter_data['iteration']
                st.markdown(f"**Iteration {iter_num}:**")
                
                # Response
                with st.container():
                    st.markdown("_Response:_")
                    st.text(iter_data['response'][:500] + "..." if len(iter_data['response']) > 500 else iter_data['response'])
                
                # Tool calls
                if iter_data.get('tool_calls'):
                    st.markdown("_Tool calls:_")
                    for tc in iter_data['tool_calls']:
                        tool_col, result_col = st.columns([1, 2])
                        
                        with tool_col:
                            st.code(f"{tc['name']}", language="python")
                        
                        # Tool results
                        if iter_data.get('tool_call_results'):
                            for result in iter_data['tool_call_results']:
                                if result['name'] == tc['name'] and result.get('id') == tc.get('id'):
                                    with result_col:
                                        result_text = result.get('result', 'No result')
                                        if len(result_text) > 200:
                                            st.text(result_text[:200] + "...")
                                        else:
                                            st.text(result_text)
                
                st.divider()

# Main UI
st.title("🤖 iExplain")
st.markdown("Agentic AI analysis framework")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Run Query", "Upload Files", "View Logs", "Settings"])

# Tab 1: Run Query
with tab1:
    st.header("Run Query")
    
    # Query input
    query = st.text_area(
        "Enter your query:",
        height=100,
        placeholder="Example: Analyze the log file and find errors..."
    )
    
    # Run button
    if st.button("Run", type="primary"):
        if not query:
            st.error("Please enter a query")
        else:
            # Get config from session state (set in Settings tab)
            provider = st.session_state.get('provider', config.get('provider', 'anthropic'))
            model = st.session_state.get('model', config.get('model', 'claude-sonnet-4-20250514'))
            api_key = st.session_state.get('api_key', config.get('api_key'))
            temperature = st.session_state.get('temperature', config.get('temperature', 0.0))
            max_tokens = st.session_state.get('max_tokens', config.get('max_tokens', 16384))
            
            if not api_key:
                st.error(f"Please set API key in Settings tab")
            else:
                with st.spinner("Running..."):
                    try:
                        # Initialize supervisor
                        supervisor = init_supervisor(provider, model, api_key, temperature, max_tokens)
                        
                        # Run query
                        result = supervisor.run(query)
                        
                        st.session_state.last_result = result
                        
                        # Display result
                        st.success("✓ Complete")
                        st.markdown("**Result:**")
                        st.write(result['content'])
                        
                        # Show log file location
                        log_files = get_log_files()
                        if log_files:
                            st.info(f"Log saved: {log_files[0].name}")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # Show last result if available
    if st.session_state.last_result:
        st.divider()
        st.markdown("### Last Result")
        st.write(st.session_state.last_result['content'])

# Tab 2: Upload Files
with tab2:
    st.header("Upload Files")
    st.markdown("Upload files to the workspace data directory")
    
    data_dir = Path(config["workspace_data"])
    st.info(f"Files will be uploaded to: `{data_dir}`")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Upload"):
            data_dir.mkdir(parents=True, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = data_dir / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✓ Uploaded: {uploaded_file.name}")
    
    # Show existing files
    st.divider()
    st.markdown("### Files in workspace:")
    
    if data_dir.exists():
        files = list(data_dir.glob("*"))
        if files:
            for f in files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {f.name}")
                # with col2:
                #     if st.button("Delete", key=f"del_{f.name}"):
                #         f.unlink()
                #         st.rerun()
        else:
            st.info("No files uploaded yet")
    else:
        st.info("Workspace directory doesn't exist yet")

# Tab 3: View Logs
with tab3:
    st.header("View Logs")
    
    log_files = get_log_files()
    
    if not log_files:
        st.info("No logs yet. Run a query first!")
    else:
        # Log selector
        log_options = [f.name for f in log_files]
        selected_log = st.selectbox("Select log:", log_options)
        
        if selected_log:
            log_file = st.session_state.logs_dir / selected_log
            log = load_log(log_file)
            
            st.divider()
            display_log(log)

# Tab 4: Settings
with tab4:
    st.header("Settings")
    
    # Provider selection
    provider = st.selectbox(
        "Provider:",
        ["anthropic", "openai", "ollama"],
        index=["anthropic", "openai", "ollama"].index(
            st.session_state.get('provider', config.get('provider', 'anthropic'))
        )
    )
    st.session_state.provider = provider
    
    # Model
    if provider == "anthropic":
        default_model = "claude-sonnet-4-20250514"
        models = [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-5-sonnet-20241022"
        ]
    elif provider == "openai":
        default_model = "gpt-4o-mini"
        models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    else:  # ollama
        # Get models from ollama installation by running `ollama list`
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            models = [line.split(" ")[0] for line in result.stdout.splitlines()[1:]]
        else:
            models = []
    
    model = st.selectbox(
        "Model:",
        models,
        index=0 if st.session_state.get('model') not in models else models.index(st.session_state.get('model'))
    )
    st.session_state.model = model
    
    # API Key
    if provider in ["anthropic", "openai"]:
        api_key = st.text_input(
            "API Key:",
            type="password",
            value=st.session_state.get('api_key', config.get('api_key', '')),
            help=f"Your {provider.title()} API key"
        )
        st.session_state.api_key = api_key
    
    # Temperature
    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.get('temperature', config.get('temperature', 0.0)),
        step=0.1
    )
    st.session_state.temperature = temperature
    
    # Max tokens
    max_tokens = st.number_input(
        "Max tokens:",
        min_value=1024,
        max_value=32768,
        value=st.session_state.get('max_tokens', config.get('max_tokens', 16384)),
        step=1024
    )
    st.session_state.max_tokens = max_tokens
    
    # Workspace info
    st.divider()
    st.markdown("### Workspace Paths:")
    st.code(f"""
Workspace: {config['workspace']}
Data:      {config['workspace_data']}
Tools:     {config['workspace_tools']}
Agents:    {config['workspace_agents']}
Logs:      ./logs
""")

# Sidebar
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    **iExplain** is a agentic framework for analyzing logs and processes.
    
    Features:
    - Configure LLM provider
    - Run agent queries
    - Upload files
    - View execution logs
    """)
    
    st.divider()
    
    # Quick stats
    log_files = get_log_files()
    data_dir = Path(config["workspace_data"])
    data_files = list(data_dir.glob("*")) if data_dir.exists() else []
    
    st.metric("Total Logs", len(log_files))
    st.metric("Uploaded Files", len(data_files))
