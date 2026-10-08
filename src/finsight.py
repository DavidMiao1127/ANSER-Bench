"""FinSight closed-corpus baseline (GPL-3.0)."""
from __future__ import annotations
"""Base agent with save/load/run capabilities."""
from typing import Dict, Any, Union, Optional, Type
import sys
import os
import pickle
import dill
import uuid
import re
import asyncio
from datetime import datetime

from src.tools.finsight_tools import get_tool_by_name
from src.common.finsight_logger import get_logger
from src.tools.finsight_tools import Tool, closed_corpus_tools


_AGENT_REGISTRY: Dict[str, Type['BaseAgent']] = {}

def register_agent_class(agent_class: Type['BaseAgent']):
    _AGENT_REGISTRY[agent_class.AGENT_NAME] = agent_class
    return agent_class

class BaseAgent:
    AGENT_NAME = 'base'
    AGENT_DESCRIPTION = 'base agent'
    NECESSARY_KEYS = ['task']
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'AGENT_NAME') and cls.AGENT_NAME != 'base':
            register_agent_class(cls)
    
    def __init__(
        self, 
        config: Any, 
        tools: list[Union[Tool, 'BaseAgent']],
        use_llm_name: str = "deepseek-chat",
        enable_code = True,
        memory = None,
        agent_id: str = None,
    ):
        self.config = config
        self.name = self.AGENT_NAME
        self.type = f'agent_{self.AGENT_NAME}'
        if agent_id is None:
            self.id = f'agent_{self.AGENT_NAME}_{uuid.uuid4().hex[:8]}'
        else:
            self.id = agent_id
        self.state = None
        self.working_dir = os.path.join(self.config.working_dir, 'agent_working', self.id)
        os.makedirs(self.working_dir, exist_ok=True)
        self.cache_dir = os.path.join(self.working_dir, '.cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.enable_code = enable_code
        if self.enable_code:
            self.executor_path = os.path.join(self.working_dir, '.executor_cache')
            os.makedirs(self.executor_path, exist_ok=True)
            self.code_executor = AsyncCodeExecutor(self.executor_path)
            self.executor_state_path = os.path.join(self.executor_path, 'state.dill')
        
        self.use_llm_name = use_llm_name
        self.llm = self.config.llm_dict[use_llm_name]
        self.memory = memory
        
        if tools is None or tools == []:
            self._set_default_tools()
        else:
            self.tools = tools
        for tool in self.tools:
            self.memory.add_dependency(tool.id, self.id)
        self.current_task_data = {}
        self.current_checkpoint = {}
        self._resume_state: Dict[str, Any] | None = None
        self.current_round = 0
        
        # Initialize logger and set agent context
        self.logger = get_logger()
        self.logger.set_agent_context(self.id, self.AGENT_NAME)
    
    def _set_default_tools(self):
        return []

    def _get_persist_extra_state(self) -> Dict[str, Any]:
        """Hook for subclasses to persist additional state."""
        return {}

    def _load_persist_extra_state(self, state: Dict[str, Any]):
        """Hook for subclasses to restore extra state."""
        return
    
    @classmethod
    async def from_checkpoint(
        cls,
        config: Any,
        memory,
        agent_id: str,
        checkpoint_name: str = 'latest.pkl',
        tools: Optional[list] = None,
        restored_agents: Optional[Dict[str, 'BaseAgent']] = None,
        **kwargs
    ) -> Optional['BaseAgent']:
        """Restore an agent from a checkpoint."""
        if restored_agents is None:
            restored_agents = {}
        
        # Return immediately if already restored
        if agent_id in restored_agents:
            return restored_agents[agent_id]
        
        # Build checkpoint path
        working_dir = os.path.join(config.working_dir, 'agent_working', agent_id)
        cache_dir = os.path.join(working_dir, '.cache')
        checkpoint_path = os.path.join(cache_dir, checkpoint_name)
        
        logger = get_logger()
        
        if not os.path.exists(checkpoint_path):
            # Try alternative checkpoints
            if not os.path.exists(cache_dir):
                logger.warning(
                    f"Cache directory not found for agent {agent_id}: {cache_dir}"
                )
                return None
            
            other_checkpoints = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
            if other_checkpoints:
                checkpoint_path = os.path.join(cache_dir, other_checkpoints[0])
                logger.info(
                    f"Using alternative checkpoint for agent {agent_id}: "
                    f"{checkpoint_path} (requested: {checkpoint_name})"
                )
            else:
                logger.warning(
                    f"No checkpoint file found for agent {agent_id}: "
                    f"requested={checkpoint_name}, cache_dir={cache_dir}, "
                    f"available_files={os.listdir(cache_dir) if os.path.exists(cache_dir) else 'N/A'}"
                )
                return None
        
        # Load checkpoint
        try:
            with open(checkpoint_path, 'rb') as f:
                state = dill.load(f)
        except Exception as e1:
            try:
                with open(checkpoint_path, 'rb') as f:
                    state = pickle.load(f)
            except Exception as e2:
                logger.error(
                    f"Failed to load checkpoint for agent {agent_id}: "
                    f"path={checkpoint_path}, "
                    f"dill_error={type(e1).__name__}: {e1}, "
                    f"pickle_error={type(e2).__name__}: {e2}"
                )
                return None
        
        agent_name = state.get('agent_name')
        if not agent_name:
            logger.warning(
                f"Checkpoint for agent {agent_id} has no agent_name: "
                f"path={checkpoint_path}"
            )
            return None
        
        # Lookup agent class from registry
        agent_class = _AGENT_REGISTRY.get(agent_name)
        if not agent_class:
            logger.warning(
                f"Agent class not found in registry for agent {agent_id}: "
                f"agent_name={agent_name}, "
                f"available_agents={list(_AGENT_REGISTRY.keys())}"
            )
            return None
        
        # Restore dependent tools if not provided
        if tools is None:
            tools = await cls._restore_tools_from_checkpoint(
                config, memory, state, checkpoint_name, restored_agents, **kwargs
            )
        
        # Restore persisted parameters
        checkpoint_use_llm_name = state.get('use_llm_name', 'deepseek-chat')
        use_llm_name = kwargs.get('use_llm_name', checkpoint_use_llm_name)
        
        # Restore additional init kwargs
        init_params = state.get('init_params', {})
        for key, value in init_params.items():
            if key not in kwargs:
                kwargs[key] = value
        
        # Instantiate agent
        agent = agent_class(
            config=config,
            memory=memory,
            agent_id=agent_id,
            use_llm_name=use_llm_name,
            tools=tools,
            **{k: v for k, v in kwargs.items() if k != 'use_llm_name'}
        )
        
        # Restore runtime state
        agent._resume_state = state
        agent.current_checkpoint = state
        agent.current_task_data = state.get('current_task_data', {})
        
        # Ensure logger context is configured
        agent.logger.set_agent_context(agent.id, agent.AGENT_NAME)
        
        # Restore executor state
        if agent.enable_code and os.path.exists(agent.executor_state_path):
            try:
                with open(agent.executor_state_path, 'rb') as ef:
                    exec_state = ef.read()
                agent.code_executor.load_state(exec_state)
            except Exception as e:
                agent.logger.error(f"Failed to load code-executor state: {e}", exc_info=True)
        
        # Restore subclass-specific state
        agent._load_persist_extra_state(state)
        
        restored_agents[agent_id] = agent
        return agent
    
    @classmethod
    async def _restore_tools_from_checkpoint(
        cls,
        config: Any,
        memory,
        state: Dict[str, Any],
        checkpoint_name: str,
        restored_agents: Dict[str, 'BaseAgent'],
        **kwargs
    ) -> list:
        """Restore tools from checkpoint state."""
        from src.tools.finsight_tools import get_tool_by_name
        
        tool_dependencies = state.get('tool_dependencies', [])
        restored_tools = []
        logger = get_logger()
        
        for dep in tool_dependencies:
            if dep['type'] == 'agent':
                # Recursively restore dependent agents
                # First, attempt to load the dependency's checkpoint to fetch init_params
                dep_agent_id = dep['agent_id']
                dep_working_dir = os.path.join(config.working_dir, 'agent_working', dep_agent_id)
                dep_cache_dir = os.path.join(dep_working_dir, '.cache')
                dep_checkpoint_path = os.path.join(dep_cache_dir, checkpoint_name)
                
                # If the specified checkpoint is missing, fall back to any available one
                if not os.path.exists(dep_checkpoint_path):
                    if os.path.exists(dep_cache_dir):
                        other_checkpoints = [f for f in os.listdir(dep_cache_dir) if f.endswith('.pkl')]
                        if other_checkpoints:
                            dep_checkpoint_path = os.path.join(dep_cache_dir, other_checkpoints[0])
                
                # Read dependency checkpoint to fetch its parameters
                dep_kwargs = {}
                if os.path.exists(dep_checkpoint_path):
                    try:
                        with open(dep_checkpoint_path, 'rb') as f:
                            dep_state = dill.load(f)
                        dep_init_params = dep_state.get('init_params', {})
                        # Only pass supported parameters; always forward use_llm_name (can be overridden)
                        dep_kwargs['use_llm_name'] = kwargs.get('use_llm_name', dep_init_params.get('use_llm_name'))
                        # Forward additional options if present in the dependency checkpoint
                        for key in ['use_vlm_name', 'use_embedding_name', 'enable_code']:
                            if key in dep_init_params:
                                dep_kwargs[key] = dep_init_params[key]
                    except Exception as e:
                        logger.warning(
                            f"Failed to load dependency agent checkpoint for {dep_agent_id}: {e}, "
                            f"using default kwargs"
                        )
                        # On failure, fall back to common kwargs
                        dep_kwargs['use_llm_name'] = kwargs.get('use_llm_name')
                else:
                    # Use generic kwargs if no checkpoint exists
                    dep_kwargs['use_llm_name'] = kwargs.get('use_llm_name')
                
                dep_agent = await cls.from_checkpoint(
                    config=config,
                    memory=memory,
                    agent_id=dep_agent_id,
                    checkpoint_name=checkpoint_name,
                    restored_agents=restored_agents,
                    **dep_kwargs
                )
                if dep_agent:
                    restored_tools.append(dep_agent)
            elif dep['type'] == 'tool':
                # Recreate tool instance
                tool_instance = get_tool_by_name(dep['tool_name'])()
                if tool_instance.id != dep['tool_id']:
                    tool_instance.id = dep['tool_id']
                restored_tools.append(tool_instance)
        
        return restored_tools

    async def save(self, state: Dict[str, Any] | None = None, checkpoint_name: str = 'latest.pkl'):
        """Persist the current agent state to a checkpoint."""
        # Capture tool dependencies (agent/tool identifiers)
        tool_dependencies = []
        for tool in self.tools:
            if isinstance(tool, BaseAgent):
                tool_dependencies.append({
                    'type': 'agent',
                    'agent_name': tool.AGENT_NAME,
                    'agent_id': tool.id,
                })
            elif isinstance(tool, Tool):
                tool_dependencies.append({
                    'type': 'tool',
                    'tool_name': tool.name,
                    'tool_id': tool.id,
                })
        
        # Persist relevant initialization parameters
        init_params = {}
        for key in ['use_llm_name', 'enable_code', 'use_vlm_name', 'use_embedding_name']:
            if hasattr(self, key):
                init_params[key] = getattr(self, key)
        
        checkpoint = {
            'agent_name': self.AGENT_NAME,
            'agent_id': self.id,
            'use_llm_name': self.use_llm_name,
            'current_time': self.current_time,
            'current_task_data': getattr(self, 'current_task_data', {}),
            'tool_dependencies': tool_dependencies,
            'init_params': init_params,
        }

        if state:
            self.current_checkpoint.update(state)
        checkpoint.update(self.current_checkpoint)
        target_path = os.path.join(self.cache_dir, checkpoint_name)
        tmp_path = target_path + '.tmp'
        try:
            with open(tmp_path, 'wb') as f:
                dill.dump(checkpoint, f)
            os.replace(tmp_path, target_path)
        except Exception:
            with open(tmp_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            os.replace(tmp_path, target_path)

        # Save code-executor state
        if self.enable_code and hasattr(self, 'code_executor'):
            try:
                state_bytes = self.code_executor.save_state()
                with open(self.executor_state_path, 'wb') as ef:
                    ef.write(state_bytes)
            except Exception as e:
                self.logger.error(f"Failed to save code-executor state: {e}", exc_info=True)

    async def load(self, checkpoint_name: str = 'latest.pkl') -> Dict[str, Any] | None:
        """Load state from a checkpoint."""
        target_path = os.path.join(self.cache_dir, checkpoint_name)
        if not os.path.exists(target_path):
            return None
        try:
            with open(target_path, 'rb') as f:
                state = dill.load(f)
        except Exception:
            with open(target_path, 'rb') as f:
                state = pickle.load(f)
        self.state = state
        # Restore essential fields
        self.current_task_data = state.get('current_task_data', {})
        # Restore code-executor state
        if self.enable_code and hasattr(self, 'code_executor') and os.path.exists(self.executor_state_path):
            try:
                with open(self.executor_state_path, 'rb') as ef:
                    exec_state = ef.read()
                self.code_executor.load_state(exec_state)
                # Ensure helper functions are re-registered
                self.code_executor.set_variable("call_tool", self._agent_tool_function)
            except Exception as e:
                self.logger.error(f"Failed to load code-executor state: {e}", exc_info=True)
        return state

    async def _prepare_executor(self):
        if self.enable_code:
            self.code_executor.set_variable("call_tool", self._agent_tool_function)

    async def _prepare_init_prompt(self, input_data: dict) -> list[dict]:
        raise NotImplementedError
    

    def _agent_tool_function(self, tool_name: str = None, **kwargs):
        """Execute a tool by name.

        This function is injected into the code-executor's global namespace so
        that LLM-generated code can call tools synchronously via
        ``call_tool(tool_name='...', **kwargs)``.

        Because it runs inside ``exec()`` which is already nested in a running
        asyncio event loop, we must **not** call ``asyncio.run()`` (that would
        deadlock or raise ``RuntimeError``).  Instead we delegate to an
        ``AsyncBridge`` that owns a separate background event loop.
        """
        from src.utils.async_bridge import get_async_bridge

        if tool_name is None:
            raise ValueError("tool_name is required")
        target_tool = None
        for tool in self.tools:
            if isinstance(tool, Tool):
                if tool.name == tool_name:
                    target_tool = tool
                    break
            elif isinstance(tool, BaseAgent):
                if tool.AGENT_NAME == tool_name:
                    target_tool = tool
                    break
        if target_tool is None:
            self.logger.warning(f"No available tools for tool_name: {tool_name}")
            self.memory.add_log(self.id, None, kwargs, [], error=True, note=f"No available tools for tool_name: {tool_name}")
            return []

        bridge = get_async_bridge()

        # Apply rate limiting if the config provides a limiter
        rate_limiter = getattr(self.config, 'rate_limiter', None)
        if rate_limiter is not None:
            # Determine service category for rate limiting
            service = "financial_apis"  # default
            tool_type = getattr(target_tool, 'type', '')
            tool_name_lower = (tool_name or '').lower()
            if 'search' in tool_name_lower or 'web' in tool_name_lower:
                service = "search_engines"
            elif 'fred' in tool_name_lower:
                service = "fred_api"
            elif 'us' in tool_name_lower and 'fred' not in tool_name_lower:
                service = "yfinance"
            try:
                bridge.run_async(rate_limiter.acquire(service))
            except Exception as rl_err:
                self.logger.debug(f"Rate limiter acquire for {service}: {rl_err}")

        try:
            if issubclass(type(target_tool), BaseAgent):
                if 'task' not in kwargs:
                    kwargs['task'] = self.current_task_data['task']
                response = bridge.run_async(target_tool.async_run(input_data=kwargs))
                response = response['final_result']
                self.memory.add_log(target_tool.id, target_tool.type, kwargs, response, error=False, note=f"Tool {target_tool.name} executed successfully")
                return response
            elif issubclass(type(target_tool), Tool):
                response = bridge.run_async(target_tool.api_function(**kwargs))
                sources = [item.source for item in response]
                data_list = [item.data for item in response]
                sources = "\n".join(sources)
                import sys
                display_note = f"[Tool Result Overview] Gather {len(response)} Tool Results.\n"
                for i, item in enumerate(response):
                    display_note += f"-{i}. Name: {item.name}\nSource: {item.source}\n"
                print(f"\n\n{display_note}\n\n", file=sys.stdout, flush=True)

                self.memory.add_log(target_tool.id, target_tool.type, kwargs, response, error=False, note=f"Tool {target_tool.name} executed successfully")
                return data_list
            else:
                self.logger.warning(f"Unknown tools: {tool_name}")
                self.memory.add_log(self.id, target_tool.type, kwargs, [], error=True, note=f"Unknown tools: {tool_name}")
                return []
        except Exception as e:
            self.logger.error(f"Tool {tool_name} execution failed: {e}", exc_info=True)
            self.memory.add_log(self.id, target_tool.type, kwargs, [], error=True, note=f"Tool {tool_name} executed failed: {e}")
            return []
    
    def _get_api_descriptions(self) -> str:
        desc = 'The usage of tool calling: `tool_result = call_tool(tool_name=\'tool_name\', **kwargs)`. (you can use custom variable names for the tool result)\n\n'
        desc += 'Below are the tools and their descriptions:\n\n'
        for tool in self.tools:
            if issubclass(type(tool), Tool):
                desc += f"- Tool: {tool.name}\nDescription: {tool.description}\nParameters: {tool.parameters}\n\nOutput: "
            elif issubclass(type(tool), BaseAgent):
                desc += f"- Tool: {tool.AGENT_NAME}\nDescription: {tool.AGENT_DESCRIPTION}\n\n"
        desc += 'The result of each tool is a varaible, please use `print` to print the result.'
        return desc
    
    def _check_necessary_data(self, input_data):
        check_keys = self.NECESSARY_KEYS
        for key in check_keys:
            if key not in input_data:
                self.logger.warning(f"{key} not in input_data")

    async def async_run(
        self, 
        input_data: dict, 
        max_iterations: int = 10,
        stop_words: list[str] = ["</execute>", "</final_result>"],
        echo=False,
        resume: bool = False,
        checkpoint_name: str = 'latest.pkl',
        prompt_function=None,
    ) -> dict:
        """Main execution loop."""
        # Ensure logger context is set (important for asyncio execution)
        self.logger.set_agent_context(self.id, self.AGENT_NAME)
        
        self._check_necessary_data(input_data)
        self.current_task_data = input_data
        await self._prepare_executor()

        # Restore or initialize conversation state
        conversation_history: list[dict]
        current_round: int
        if prompt_function is None:
            prompt_function = self._prepare_init_prompt
        if resume:
            state = await self.load(checkpoint_name=checkpoint_name)
            if state is not None:
                conversation_history = state.get('conversation_history', [])
                current_round = int(state.get('current_round', 0))
                if 'return_dict' in state:
                    return state['return_dict']
            else:
                conversation_history = await prompt_function(input_data)
                current_round = 0
        else:
            conversation_history = await prompt_function(input_data)
            current_round = 0
    
        while current_round < max_iterations+1:
            self.logger.info(f"Iteration {current_round + 1}")
            current_round += 1
            self.current_round = current_round
            response = await self.llm.generate(messages = conversation_history, stop=stop_words)
            action_type, action_content = self._parse_llm_response(response)
            if echo:
                self.logger.info(f"LLM response this step: {response}")
                self.logger.info("--------")
            # Execute asynchronously
            action_result = await self._execute_action(action_type, action_content)
            action_result['llm_response'] = response
            if echo:
                self.logger.info(f"Action result this step: {action_result['result']}")
                self.logger.info("--------")
            conversation_history.append({"role": "assistant", "content": action_result['llm_response']})
            conversation_history.append({"role": "user", "content": action_result['result']})
            self.logger.debug("--Begin of Execution Result--")
            self.logger.debug(action_result['result'])
            self.logger.debug("--End of Execution Result--")

            # Save each iteration to support resume
            current_state = {
                'conversation_history': conversation_history,
                'current_round': current_round,
                'input_data': input_data,
                'stop_words': stop_words,
            }
            current_state.update(self._get_persist_extra_state())
            self.state = current_state
            await self.save(
                state=current_state,
                checkpoint_name=checkpoint_name,
            )
            
            if not action_result['continue']:
                break
        
        return_dict = {}
        if current_round >= max_iterations and action_result['continue']:
            # Hit iteration limit; fall back to summary handler
            return_dict = await self._handle_max_round(conversation_history)
        else:
            return_dict = {
                'conversation_history': conversation_history,
                'final_result': action_result['result'],
            }
        return_dict['input_data'] = input_data
        return_dict['working_dir'] = self.working_dir
        # Save final state before exiting
        current_state = {
            'conversation_history': conversation_history,
            'current_round': current_round,
            'input_data': input_data,
            'stop_words': stop_words,
            'return_dict': return_dict,
        }
        current_state.update(self._get_persist_extra_state())
        await self.save(
            state=current_state,
            checkpoint_name=checkpoint_name,
        )
        self.memory.save()
        
        return return_dict

    async def _handle_max_round(self, conversation_history):
        return {'coversation_history': conversation_history, 'final_result': conversation_history[-1]['content']}

    def _parse_llm_response(self, response: str) -> tuple[str, str]:
        """Parse the LLM response to extract action tags."""
        response = response.replace("<thinking>", "\n").replace("</thinking>", "\n")
        response = response.replace("<think>", "\n").replace("</think>", "\n")
        pattern = re.compile(r"<([\w_]+)>(.*?)</\1>", re.DOTALL)
        matches = list(pattern.finditer(response))
        
        if not matches:
            return "final", response
        match = matches[-1]

        tag_name = match.group(1)
        if tag_name == 'execute':
            tag_name = 'code'
        if tag_name == 'final_result':
            tag_name = 'final'
        content_string = match.group(2).strip()  # Remove surrounding whitespace

        return tag_name, content_string

    
    async def _execute_action(self, action_type: str, action_content: str):
        handler_method_name = f"_handle_{action_type}_action"
        handler = getattr(self, handler_method_name, None)

        if handler and callable(handler):
            return await handler(action_content)
        else:
            return await self._handle_default_action(action_type, action_content)
    

    async def _handle_code_action(self, action_content: str):
        code_result = await self.code_executor.execute(code=action_content)
        code_result = self._format_execution_result(code_result)
        return {
            "action": "generate_code",
            "action_content": action_content,
            "result": code_result,
            "continue": True,
        }

    async def _handle_final_action(self, action_content: str):
        return {
            "action": "final_result",
            "action_content": action_content,
            "result": action_content,
            "continue": False,
        }

    async def _handle_default_action(self, action_type: str, action_content: str):
        return {
            "action": "invalid_response",
            "action_content": action_content,
            "result": f"Unknown action_type '{action_type}'. Please respond using the required XML tags.",
            "continue": True,
        }
     
        
    def _format_execution_result(self, result: Dict[str, Any]) -> str:
        feedback = []

        if result["error"] is False:
            feedback.append("Code execution: success\n")

            if result["stdout"]:
                feedback.append(f"Console output:\n{result['stdout']}\n\n")

            if result.get("variables"):
                feedback.append("New variables:")
                for var_name, var_info in result["variables"].items():
                    feedback.append(f"  - {var_name}: {var_info}")
            if result.get("additional_notes"):
                feedback.append(f"Additional notes: {result['additional_notes']}\n")
        else:
            feedback.append("Code execution: failed\n")
            if result["stderr"]:
                feedback.append(f"Error message: {result['stderr']}\n")
            if result["stdout"]:
                feedback.append(f"Partial output: {result['stdout']}\n")
        return "\n".join(feedback)    
        
        
        


from typing import List, Dict, Any, Tuple




from src.tools.finsight_tools import ToolResult

class DeepSearchAgent(BaseAgent):
    AGENT_NAME = 'deepsearch agent'
    AGENT_DESCRIPTION = (
        "Tool: Deep Search\n"
        "Description: run comprehensive web searches (news, filings, research, etc.) "
        "to gather evidence for a given task.\n"
        "Parameters: query:str (describe exactly what information is needed; "
        "avoid loose keyword lists).\n"
    )
    NECESSARY_KEYS = ['task', 'query']
    def __init__(
        self,
        config,
        tools = None,
        use_llm_name: str = "deepseek-chat",
        enable_code = False,
        memory = None,
        agent_id: str = None
    ):
        # Use search + click tools directly; no code interpreter required
        if tools is None:
            raise ValueError('Closed-corpus search and click tools are required')
        super().__init__(
            config=config,
            tools=tools,
            use_llm_name=use_llm_name,
            enable_code=enable_code,
            memory=memory,
            agent_id=agent_id
        )
        # Load prompts using the new YAML-based loader
        from src.common.finsight_prompt import get_prompt_loader
        
        self.prompt_loader = get_prompt_loader('search_agent', report_type='general')
        self.DEEP_SEARCH_PROMPT = self.prompt_loader.get_prompt('deep_search')
        self.link2name = {}
        
        # Track all valid links from search results for validation
        self.valid_links = {}  # {url: {title, description, query}}
        # Track sources actually used (clicked/browsed)
        self.used_sources = {}  # {url: {title, content_summary}}
    
    
    async def _prepare_init_prompt(self, input_data: dict) -> list[dict]:
        basic_task = input_data.get('task', '')
        query = input_data.get('query', None)
        max_iterations = input_data.get('max_iterations', 5)

        if not query:
            raise ValueError("Input data must contain a 'task' key.")
        
        # Get target language from config
        target_language = self.config.config.get('language', 'zh')
        language_mapping = {
            'zh': 'Chinese (中文)',
            'en': 'English'
        }
        target_language_name = language_mapping.get(target_language, target_language)
            
        return [{
            "role": "user",
            "content": self.DEEP_SEARCH_PROMPT.format(
                basic_task=basic_task,
                question=query,
                current_time=self.current_time,
                max_iterations=max_iterations,
                target_language=target_language_name
            )
        }]

    async def _handle_max_round(self, conversation_history):
        conversation_history = [item["content"] for item in conversation_history]
        analysis_info = "\n\n".join(conversation_history)
        prompt = f"You have reached the maximum number of running iterations. Directly give the summary of your search process based on the conversation history.\n\nConversation history: {analysis_info}\n\n"
        response = await self.llm.generate(
            messages = [
                {"role": "user", "content": prompt}
            ],
            response_format = {"type": "json_object"}
        )
        final_result = response
        return {'coversation_history': conversation_history, 'final_result': final_result}
    
    async def _handle_search_action(self, action_content):
        search_engine = [item for item in self.tools if 'search' in item.name.lower()][0]
        self.logger.info(f"Search action started: query={action_content}")
        try:
            search_result = await search_engine.api_function(action_content)
            search_result_list = []
            if len(search_result) == 0:
                result = f"Query `{action_content}` returned no results; please try again."
            else:
                result = f"Search results for `{action_content}`\n"
 
                for idx, item in enumerate(search_result):
                    title = item.name
                    link = item.link
                    description = item.description
                    search_result_list.append({
                        'query': action_content,
                        'title': title,
                        'link': link,
                        'description': description
                    })
                    self.link2name[link] = title
                    # Track this as a valid link for later validation
                    self.valid_links[link] = {
                        'title': title,
                        'description': description,
                        'query': action_content
                    }
                    result += 'Result ' + str(idx + 1) + ':\n'
                    result += f"Title: {title}\n"
                    result += f"Link: {link}\n"
                    result += f"Summary: {description}\n\n"
                    
            for search_item in search_result:
                self.memory.add_data(search_item)
            self.memory.add_log(
                id = search_engine.id, 
                type=search_engine.type,
                input_data = {'query': action_content}, 
                output_data = {'result': search_result_list}, 
                error=False, 
                note=f"Search engine {search_engine.name} executed successfully"
            )
            self.logger.info(f"Search action done: query={action_content}")
                
        except Exception as e:
            result = f"Query `{action_content}` failed with error: {str(e)}. Please retry."
            self.memory.add_log(
                id = search_engine.id, 
                type=search_engine.type,
                input_data = {'query': action_content}, 
                output_data = {"result": result}, 
                error=True, 
                note=f"Search engine {search_engine.name} executed failed: {str(e)}"
            )
            self.logger.error(f"Search action failed: query={action_content}, error={e}", exc_info=True)
        
        # On the last iteration, append available sources reminder
        if self.current_round >= (self.max_iterations - 1):
            result += "\n\n⚠️ You have reached the maximum number of running iterations. Please provide your final report now."
            result += self._build_available_sources_list()
        return {
            "action": "search",
            "action_content": action_content,
            "result": result,
            "continue": True
        }
    
    async def _handle_click_action(self, action_content):
        click_engine = [item for item in self.tools if 'content fetcher' in item.name.lower()][0]
        current_task = self.current_task_data.get('task', '')
        query = self.current_task_data.get('query', '')

        # Validate that the URL was from search results
        if action_content not in self.valid_links:
            self.logger.warning(f"Click rejected: URL not found in search results: {action_content}")
            # Provide available links as guidance
            available_links_hint = ""
            if self.valid_links:
                available_links_hint = "\n\nAvailable links from search results:\n"
                for idx, (url, info) in enumerate(list(self.valid_links.items())[:10], 1):
                    available_links_hint += f"{idx}. {info['title']}\n   URL: {url}\n"
            
            result = (
                f"ERROR: The URL '{action_content}' was not found in your search results. "
                f"You can ONLY click URLs that appeared in previous search results. "
                f"Please use one of the URLs from your search results, or perform a new search."
                f"{available_links_hint}"
            )
            return {
                "action": "click",
                "action_content": action_content,
                "result": result,
                "continue": True
            }

        try:
            self.logger.info(f"Click action started: url={action_content}")
            click_result = await click_engine.api_function([action_content], f'Research goal: {current_task}; query: {query}')
            if len(click_result) == 0:
                result = "Failed to fetch content for url: " + action_content
            else:
                result = click_result[0].data
                # Track this as a used source with content summary
                source_title = self.link2name.get(action_content, self.valid_links.get(action_content, {}).get('title', 'Unknown'))
                self.used_sources[action_content] = {
                    'title': source_title,
                    'content_preview': result[:500] if len(result) > 500 else result
                }
            # add to memory
            if click_result[0].link in self.link2name:
                click_result[0].name = self.link2name[click_result[0].link]
            if not ('error' in click_result[0].name.lower()):
                self.memory.add_data(click_result[0])
            self.memory.add_log(
                id = click_engine.id, 
                type=click_engine.type,
                input_data = {'url': action_content}, 
                output_data = {"result": result}, 
                error=False, 
                note=f"Click engine {click_engine.name} executed successfully"
            )
            self.logger.info(f"Click action done: url={action_content}")
            
        except Exception as e:
            result =  "Failed to fetch url: " + action_content + "\n"
            result += f'Error: {e}'
            self.memory.add_log(
                id = click_engine.id, 
                type=click_engine.type,
                input_data = {'url': action_content}, 
                output_data = {"result": result}, 
                error=True, 
                note=f"Click engine {click_engine.name} executed failed: {str(e)}"
            )
            self.logger.error(f"Click action failed: url={action_content}, error={e}", exc_info=True)
        
        # On the last iteration, append available sources reminder
        if self.current_round >= (self.max_iterations - 1):
            result += "\n\n⚠️ You have reached the maximum number of running iterations. Please provide your final report now."
            result += self._build_available_sources_list()
            
        return {
            "action": "click",
            "action_content": action_content,
            "result": result,
            "continue": True
        }

    def _build_available_sources_list(self) -> str:
        """Build a formatted list of all available sources from search results and browsed pages."""
        if not self.valid_links and not self.used_sources:
            return ""
        
        sources_text = "\n\n---\n**VERIFIED SOURCES AVAILABLE FOR CITATION:**\n"
        sources_text += "(You may ONLY use URLs from this list in your References section)\n\n"
        
        # First list browsed/used sources (highest quality)
        if self.used_sources:
            sources_text += "**Sources you have browsed (recommended for citation):**\n"
            for idx, (url, info) in enumerate(self.used_sources.items(), 1):
                sources_text += f"  {idx}. {info['title']}\n"
                sources_text += f"     URL: {url}\n"
        
        # Then list search results that weren't clicked
        unclicked_sources = {url: info for url, info in self.valid_links.items() 
                           if url not in self.used_sources}
        if unclicked_sources:
            sources_text += "\n**Additional sources from search results (snippets only):**\n"
            for idx, (url, info) in enumerate(unclicked_sources.items(), 1):
                sources_text += f"  {idx}. {info['title']}\n"
                sources_text += f"     URL: {url}\n"
                sources_text += f"     Summary: {info['description'][:200]}...\n" if len(info['description']) > 200 else f"     Summary: {info['description']}\n"
        
        sources_text += "\n---\n"
        return sources_text

    async def _handle_report_action(self, action_content: str):
        """Handle a 'final' action from the LLM."""
        return {
            "action": "final_report",
            "action_content": action_content,
            "result": action_content,
            "continue": False,
        }
    
    def _get_persist_extra_state(self) -> dict:
        """Persist valid_links and used_sources for resume support."""
        return {
            'valid_links': self.valid_links,
            'used_sources': self.used_sources,
            'link2name': self.link2name,
        }
    
    def _load_persist_extra_state(self, state: dict):
        """Restore valid_links and used_sources from checkpoint."""
        self.valid_links = state.get('valid_links', {})
        self.used_sources = state.get('used_sources', {})
        self.link2name = state.get('link2name', {})

    async def async_run(
        self, 
        input_data: dict, 
        max_iterations: int = 30,
        stop_words: list[str] = [],
        echo=False,
        resume: bool = True,
        checkpoint_name: str = 'deepsearch_latest.pkl',
        # stop_words: list[str] = ["</click>", "</search>", "</report>"]
    ) -> dict:
        input_data['max_iterations'] = max_iterations
        self.max_iterations = max_iterations
        await self._prepare_executor()
        run_result = await super().async_run(
            input_data=input_data,
            max_iterations=max_iterations,
            stop_words=stop_words,
            echo=echo,
            resume=resume,
            checkpoint_name=checkpoint_name,
        )
        agent_result = DeepSearchResult(
            query=input_data['query'], 
            name=f"Summary of the search process for {input_data['query']}", 
            description=run_result['final_result'],
            data=run_result['final_result'], 
            source=self.AGENT_NAME
        )
        self.memory.add_data(agent_result)
        return run_result
        
# TODO: add agentresult class
class DeepSearchResult(ToolResult):
    def __init__(self, query, name, description, data, source=""):
        super().__init__(name, description, data, source)
        self.query = query

    def __str__(self):
        format_output = f'Summary Search Result for {self.query}\n'
        format_output += f"Summary: {self.description}\n"
        return format_output

    def __repr__(self):
        return self.__str__()

#!/usr/bin/env python3



import argparse
import asyncio
import copy
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FINSIGHT_ROOT", WORKSPACE_ROOT / "vendor/finsight"))
OUTPUT_ROOT = WORKSPACE_ROOT / "outputs/finsight"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.common.finsight_llm import AsyncLLM


class EvaluationConfig:
    def __init__(self, model: str, base_url: str, api_key: str, language: str,
                 domain: str, corpus_task_type: str, output: Path):
        self.corpus_root = WORKSPACE_ROOT / "corpus"
        self.agent_index_dir = WORKSPACE_ROOT / "indices/agents"
        self.config = {
            "language": language,
            "search_provider": "corpus",
            "offline_corpus_only": True,
            "corpus_domain": domain,
            "corpus_language": language,
            "corpus_task_type": corpus_task_type,
            "corpus_top_k": 30 if domain == "science" else 10,
        }
        safe_model = model.replace("/", "_")
        self.working_dir = str(
            output.parent / ".anser_runs" / safe_model / domain / language / corpus_task_type
        )
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        self.llm_dict = {
            model: AsyncLLM(
                base_url=base_url,
                api_key=api_key,
                model_name=model,
                generation_params={"temperature": 0, "max_tokens": 8192},
            )
        }


class EvaluationMemory:
    """Small in-memory adapter implementing the methods used by DeepSearchAgent."""

    def __init__(self):
        self.data = []
        self.logs = []
        self.dependencies = []

    def add_dependency(self, child_id, parent_id):
        self.dependencies.append((child_id, parent_id))

    def add_data(self, value):
        self.data.append(value)

    def add_log(self, *args, **kwargs):
        self.logs.append((args, kwargs))

    def save(self):
        return None


SPLITS = ("descriptive", "predictive", "prescriptive")
_API_KEY_PLACEHOLDERS = {
    "your_api_key",
    "your-api-key",
    "replace_me",
    "replace-me",
    "changeme",
    "xxx",
}


def default_output_path(model: str, domain: str, language: str, split: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "configured-model"
    return (
        OUTPUT_ROOT
        / safe_model
        / domain
        / language
        / f"{domain}_{language}_{split}.jsonl"
    )


def is_placeholder_api_key(value: str | None) -> bool:
    return not value or value.strip().casefold() in _API_KEY_PLACEHOLDERS


def resolve_api_key(model, supplied_key=None):
    from src.common.generation import load_env
    load_env()
    return supplied_key or os.getenv('GENERATION_API_KEY')



def write_results(path: Path, records: list[dict]) -> None:
    """Atomically write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def load_results(path: Path) -> list[dict]:
    """Load every JSONL record without changing its success/error status."""
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        records.append(record)
    return records


def load_successful_results(path: Path) -> list[dict]:
    """Load completed JSONL records, excluding failures so they can be retried."""
    return [record for record in load_results(path) if not record.get("error")]


def replace_result(records: list[dict], replacement: dict) -> list[dict]:
    """Replace one task result in memory without dropping any unrelated row."""
    replacement_id = replacement.get("id")
    updated = []
    inserted = False
    for record in records:
        if record.get("id") == replacement_id:
            if not inserted:
                updated.append(replacement)
                inserted = True
            continue
        updated.append(record)
    if not inserted:
        updated.append(replacement)
    return updated


def corpus_task_type(task: dict, domain: str, language: str, split: str) -> str:
    if domain == "finance" and language == "en" and split == "prescriptive":
        task_type = task.get("type")
        if task_type not in {"RegBench", "FinAuditing"}:
            raise ValueError(f"Unsupported English finance prescriptive task type: {task_type!r}")
        return task_type
    return split


def answer_schema_for(task: dict, domain: str, language: str, split: str) -> dict | None:
    """Return a value-free schema for strict structured-answer tasks."""
    if (
        domain == "finance"
        and language == "en"
        and split == "prescriptive"
        and task.get("type") == "FinAuditing"
    ):
        return {
            "type": "object",
            "keys": ["applied_rule", "expected_value"],
            "properties": {
                "applied_rule": {"type": "string", "non_empty": True},
                "expected_value": {"type": "number"},
            },
        }
    if not (domain == "legal" and language in {"zh", "en"} and split == "predictive"):
        return None
    reference = task.get("ground_truth")
    if not (
        isinstance(reference, dict)
        and list(reference) == ["option", "value"]
        and isinstance(reference.get("value"), dict)
    ):
        raise ValueError(f"Invalid legal predictive answer template for {task.get('id')}")
    value_keys = list(reference["value"])
    options = (
        ["正向关联", "负向关联", "无明确关联"]
        if language == "zh"
        else ["positive", "negative", "no_clear_relationship"]
    )
    return {
        "type": "object",
        "keys": ["option", "value"],
        "properties": {
            "option": {
                "type": "string",
                "enum": options,
            },
            "value": {
                "type": "object",
                "keys": value_keys,
                "properties": {
                    key: {"type": "integer" if key == "n" else "number"}
                    for key in value_keys
                },
            },
        },
    }


def answer_matches_schema(answer, schema: dict | None) -> bool:
    """Check exact JSON key order, nesting, and primitive value types."""
    if schema is None:
        return True
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(answer, dict) or list(answer) != schema.get("keys"):
            return False
        properties = schema.get("properties", {})
        return all(answer_matches_schema(answer[key], properties[key]) for key in schema["keys"])
    if expected_type == "string":
        if not isinstance(answer, str) or (schema.get("non_empty") and not answer.strip()):
            return False
        return answer in schema.get("enum", [answer])
    if expected_type == "integer":
        return isinstance(answer, int) and not isinstance(answer, bool)
    if expected_type == "number":
        return (
            isinstance(answer, (int, float))
            and not isinstance(answer, bool)
            and math.isfinite(answer)
        )
    return False


def regbench_answer_matches_format(answer) -> bool:
    """Require the query-mandated conclusion, application, and CFR citations."""
    if not isinstance(answer, str) or not answer.strip():
        return False
    return all((
        re.match(r"^\s*Conclusion\s*:", answer, re.IGNORECASE),
        re.search(r"(?:^|\n)\s*Rule application\s*:", answer, re.IGNORECASE),
        re.search(r"(?:^|\n)\s*Citations\s*:", answer, re.IGNORECASE),
        re.search(
            r"(?:\b12\s*C\.?F\.?R\.?\b|§\s*217(?:\.|\b)|\bsection\s+217(?:\.|\b))",
            answer,
            re.IGNORECASE,
        ),
    ))


def result_matches_task_contract(record: dict, task: dict, domain: str, language: str, split: str) -> bool:
    if record.get("error"):
        return False
    answer = record.get("answer")
    evidence = record.get("retrieved_evidence")
    if answer in (None, "", [], {}) or not isinstance(evidence, list) or not evidence:
        return False
    if (
        domain == "finance"
        and language == "en"
        and split == "prescriptive"
        and task.get("type") == "RegBench"
        and not regbench_answer_matches_format(answer)
    ):
        return False
    return answer_matches_schema(answer, answer_schema_for(task, domain, language, split))


async def run_task(task: dict, config: EvaluationConfig, model: str, domain: str,
                   language: str, split: str,
                   max_iterations: int, task_timeout: float, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        started = time.perf_counter()
        try:
            task_config = copy.copy(config)
            safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task.get("id", "unknown"))).strip("_")
            task_config.working_dir = str(Path(config.working_dir) / safe_task_id)
            Path(task_config.working_dir).mkdir(parents=True, exist_ok=True)
            task_config.llm_dict = {model: AsyncLLM(config.llm_dict[model].client.base_url, config.llm_dict[model].client.api_key, model,
                                                     config.llm_dict[model].generation_params)}
            agent = DeepSearchAgent(
                config=task_config,
                tools=closed_corpus_tools(task, task_config),
                use_llm_name=model,
                memory=EvaluationMemory(),
                enable_code=False,
            )
            result = await asyncio.wait_for(
                agent.async_run(
                    input_data={
                        "id": task["id"],
                        "query": task["query"],
                        "instruction": task["instruction"],
                        "task": task["instruction"],
                        "domain": domain,
                        "language": language,
                        "split": split,
                        "type": task.get("type"),
                        "answer_schema": answer_schema_for(task, domain, language, split),
                    },
                    max_iterations=max_iterations,
                    resume=False,
                    echo=False,
                ),
                timeout=task_timeout,
            )
            record = json.loads(result["final_result"])
            record.setdefault("id", task["id"])
            record["retrieved_evidence"] = agent.tools[0].session.evidence()
            record["total_tokens"] = agent.llm.total_tokens if not agent.llm.unknown_calls else None
            agent.tools[0].session.close()
            record["model"] = model
            record["domain"] = domain
            record["language"] = language
            record["split"] = split
            return record
        except Exception as exc:
            if "agent" in locals(): agent.tools[0].session.close()
            return {
                "id": task.get("id"),
                "answer": None,
                "retrieved_evidence": [],
                "latency_seconds": round(time.perf_counter() - started, 6),
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "model": model,
                "domain": domain,
                "language": language,
                "split": split,
                "error": f"{type(exc).__name__}: {exc}",
            }


async def async_main(args) -> int:
    if not args.skip_preflight:
        print("Checking model endpoint and API key before scheduling tasks...", flush=True)
        probe = AsyncLLM(
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model,
            generation_params={"temperature": 0, "max_tokens": 8},
        )
        try:
            await probe.generate(
                [{"role": "user", "content": "Reply with OK."}],
                max_retries_per_model=1,
                include_stop_string=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Model preflight failed for model={args.model} base_url={args.base_url}: {exc}"
            ) from exc
        print("Model preflight passed.", flush=True)

    available_splits = ("descriptive", "predictive") if args.domain == "science" else SPLITS
    requested_splits = available_splits if args.split == "all" else (args.split,)
    if args.split not in (*available_splits, "all"):
        raise ValueError(f"Split {args.split!r} is not available for domain {args.domain!r}")
    tasks = []
    for split in requested_splits:
        query_path = (
            WORKSPACE_ROOT / "query" / "science" / f"{split}.json"
            if args.domain == "science"
            else WORKSPACE_ROOT / "query" / args.domain / args.language / f"{split}.json"
        )
        tasks.extend(
            (task, split, corpus_task_type(task, args.domain, args.language, split))
            for task in json.loads(query_path.read_text(encoding="utf-8"))
        )
    output = args.output.resolve()
    full_order = {task["id"]: index for index, (task, _, _) in enumerate(tasks)}
    if args.rerun_ids:
        if not args.resume:
            raise ValueError("--rerun-ids requires --resume so non-target records are preserved")
        if not output.exists():
            raise ValueError("--rerun-ids requires an existing output JSONL file")
        requested_ids = []
        available_ids = set(full_order)
        for value in args.rerun_ids:
            identifier = value if not value.isdigit() else f"{args.domain}_{args.language}_{args.split}_{value}"
            if identifier not in available_ids:
                raise ValueError(f"Unknown task ID for the selected split: {value!r}")
            if identifier not in requested_ids:
                requested_ids.append(identifier)
        requested_set = set(requested_ids)
        selected = [spec for spec in tasks if spec[0]["id"] in requested_set]
    else:
        selected = tasks[args.offset :] if args.limit == 0 else tasks[args.offset : args.offset + args.limit]
    if not selected:
        raise ValueError("Selected task range is empty")

    order = full_order if args.rerun_ids else {
        task["id"]: index for index, (task, _, _) in enumerate(selected)
    }
    if args.rerun_ids:
        existing_records = load_results(output)
        existing_counts = Counter(
            record.get("id") for record in existing_records
            if record.get("id") in full_order
        )
        missing_ids = [identifier for identifier in full_order if not existing_counts[identifier]]
        duplicate_ids = [identifier for identifier, count in existing_counts.items() if count > 1]
        if missing_ids or duplicate_ids:
            problems = []
            if missing_ids:
                problems.append(f"missing {len(missing_ids)} task IDs")
            if duplicate_ids:
                problems.append(f"duplicated {len(duplicate_ids)} task IDs")
            raise ValueError(
                "Cannot safely perform a targeted rerun because the existing output is incomplete: "
                + ", ".join(problems)
            )
        records = existing_records
        print(
            "Preserving existing rows until each atomic replacement is ready; force-rerunning: "
            + ", ".join(requested_ids),
            flush=True,
        )
    else:
        records = load_successful_results(output) if args.resume else []
    selected_by_id = {
        task["id"]: (task, split)
        for task, split, _ in selected
    }
    resumable_records = []
    rejected_records = []
    if not args.rerun_ids:
        for record in records:
            identifier = record.get("id")
            if identifier not in order:
                continue
            task, split = selected_by_id[identifier]
            if result_matches_task_contract(record, task, args.domain, args.language, split):
                resumable_records.append(record)
            else:
                rejected_records.append(identifier)
        records = resumable_records
        if rejected_records:
            print(
                "Retrying records that violate the current answer/evidence contract: "
                + ", ".join(rejected_records),
                flush=True,
            )
    completed_ids = {
        record["id"] for record in records
        if record.get("id") in selected_by_id
    }
    if args.rerun_ids:
        completed_ids.difference_update(requested_set)
    remaining = [spec for spec in selected if spec[0]["id"] not in completed_ids]
    if completed_ids:
        print(f"Resuming with {len(completed_ids)}/{len(selected)} successful records", flush=True)

    config_keys = {(split, task_type) for _, split, task_type in remaining}
    configs = {
        key: EvaluationConfig(
            args.model, args.base_url, args.api_key, args.language, args.domain, key[1], output,
        )
        for key in config_keys
    }
    semaphore = asyncio.Semaphore(args.concurrency)
    pending = [
        run_task(task, configs[(split, task_type)], args.model, args.domain, args.language, split,
                 args.max_iterations, args.task_timeout, semaphore)
        for task, split, task_type in remaining
    ]
    finished_targets = len(selected) - len(remaining)
    for future in asyncio.as_completed(pending):
        record = await future
        records = replace_result(records, record) if args.rerun_ids else [*records, record]
        records.sort(key=lambda item: order.get(item["id"], len(selected)))
        write_results(output, records)
        finished_targets += 1
        status = "ERROR" if record.get("error") else "OK"
        print(f"[{finished_targets}/{len(selected)}] {record['id']} {status}", flush=True)

    if not pending:
        write_results(output, records)

    failures = sum(bool(record.get("error")) for record in records)
    print(f"Saved {len(records)} records to {output} ({failures} failures)")
    return 1 if failures else 0


def main() -> int:
    load_dotenv(WORKSPACE_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run corpus-only ANSER inference.")
    parser.add_argument("--domain", choices=("finance", "legal", "science"), default="finance")
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="descriptive")
    parser.add_argument("--limit", type=int, default=5, help="Number of tasks; use 0 for every selected task")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default=os.getenv("GENERATION_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("GENERATION_BASE_URL"))
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        help="Result JSONL path (default: outputs/finsight/<model>/<domain>/<language>/...).",
    )
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--task-timeout", type=float, default=900, help="Maximum wall-clock seconds per task")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Keep successful records already present in output")
    parser.add_argument(
        "--rerun-ids",
        nargs="+",
        default=[],
        metavar="ID",
        help=(
            "With --resume, force-rerun only these task IDs while preserving every other output "
            "record. Numeric suffixes such as 59 are accepted for a single selected split."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the one-request endpoint/key check before scheduling tasks",
    )
    args = parser.parse_args()
    args.api_key = resolve_api_key(args.model, args.api_key)
    if not args.model or not args.base_url:
        parser.error("model and base URL must be supplied via flags or GENERATION_MODEL/GENERATION_BASE_URL")
    if args.output is None:
        args.output = default_output_path(args.model, args.domain, args.language, args.split)
    if is_placeholder_api_key(args.api_key):
        parser.error(
            "Set GENERATION_API_KEY in .env or pass --api-key "
            "(the current value is empty or a placeholder)"
        )
    if args.output.suffix.lower() != ".jsonl":
        parser.error("output must use the .jsonl extension")
    if args.domain == "science" and args.language != "en":
        parser.error("science tasks and corpus are English-only; use --language en")
    if args.limit < 0 or args.offset < 0 or args.max_iterations < 3 or args.concurrency < 1 or args.task_timeout < 1:
        parser.error("limit/offset must be non-negative, concurrency positive, max-iterations at least 3, and task-timeout positive")
    print(f"Using model={args.model} base_url={args.base_url}", flush=True)
    return asyncio.run(async_main(args))


def run_selected(args,tasks,result_path):
    from src.common.selected_agents import generation,identity,execute,dry_run
    if args.dry_run or args.build_index_only:
        dry_run(tasks,args,result_path);return
    model,base,key=generation(args)
    async def worker(task):
        domain,language,split=identity(task)
        config=EvaluationConfig(model,base,key,language,domain,corpus_task_type(task,domain,language,split),result_path)
        config.corpus_root=args.corpus_root
        config.agent_index_dir=args.agent_index_dir
        return await run_task(task,config,model,domain,language,split,args.max_iterations or 12,args.deadline or 900,asyncio.Semaphore(1))
    asyncio.run(execute(tasks,args,result_path,worker))


if __name__ == "__main__":
    from src.common.cli import baseline_main
    baseline_main("finsight")
