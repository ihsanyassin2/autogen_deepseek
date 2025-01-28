import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from custom_user_proxy_agent import UserProxyAgent
from custom_planner_agent import PlannerAgent
from custom_gpt_researcher import GPTResearcherAgent
from custom_web_surfer import WebSurferAgent
from custom_critic_agent import CriticAgent
from custom_first_principles_analyst import FirstPrinciplesAnalyst
from custom_devils_advocate import DevilsAdvocate
from custom_critical_friend import CriticalFriend
from custom_unconventional_strategist import UnconventionalStrategist
from custom_contrarian import Contrarian
from custom_groupchat import GroupChat
from custom_groupchat_manager import GroupChatManager
from custom_conversable_agent import Agent, ConversableAgent

@dataclass
class TeamConfig:
    """Configuration for team composition."""
    use_first_principles: bool = True
    use_devils_advocate: bool = True
    use_critical_friend: bool = True
    use_unconventional_strategist: bool = True
    use_contrarian: bool = True
    max_rounds: int = 20
    temperature: float = 0.3
    allow_repeat_speaker: bool = False
    speaker_selection_method: str = "auto"
    user_proxy_mode: str = "NEVER"

def create_llm_config(temperature=0.3):
    """Create a standardized LLM configuration."""
    return {
        "config_list": [
            {
                "api_type": "ollama",
                "model": "deepseek-r1",
                "api_base": "http://localhost:8000",
                "seed": 12345,
                "allow_repeat_speaker": True,
            }
        ],
        "temperature": temperature,
        "timeout": 120,
    }

def initialize_agents(task: str, config: TeamConfig) -> List[Agent]:
    """Initialize agents with proper ConversableAgent compatibility"""
    agents = []

    # Define reply functions first to avoid lambda issues
    def user_reply_func(recipient, messages, sender=None, **kwargs):
        """Reply function for user proxy agent"""
        return False, {"role": "user", "content": "Acknowledged."}

    def assistant_reply_func(recipient, messages, sender=None, **kwargs):
        """Reply function for assistant agents"""
        return False, {"role": "assistant", "content": messages[-1].get("content", "")}

    # User Proxy Agent
    user_proxy = UserProxyAgent(
        name="User Proxy",
        human_input_mode=config.user_proxy_mode,
        code_execution_config={},
        system_message=f"User proxy for: {task}",
    )
    user_proxy.register_reply(Agent, user_reply_func)
    agents.append(user_proxy)

    # Agent creation helper
    def create_agent(cls, name, system_msg, temp_multiplier=1.0):
        agent = cls(
            name=name,
            llm_config=create_llm_config(temperature=config.temperature * temp_multiplier),
            system_message=system_msg
        )
        agent.register_reply(Agent, assistant_reply_func)
        return agent

    # Add agents for different roles
    if config.use_first_principles:
        agents.append(create_agent(
            FirstPrinciplesAnalyst,
            "First Principles Analyst",
            system_msg=f"First Principles Specialist for: {task}\n- Foundational analysis\n- Assumption validation",
        ))

    if config.use_devils_advocate:
        agents.append(create_agent(
            DevilsAdvocate,
            "Devil's Advocate",
            system_msg=f"Devil's Advocate for: {task}\n- Identify flaws\n- Explore failure scenarios",
        ))

    if config.use_critical_friend:
        agents.append(create_agent(
            CriticalFriend,
            "Critical Friend",
            system_msg=f"Critical Friend for: {task}\n- Constructive feedback\n- Supportive critique",
        ))

    if config.use_unconventional_strategist:
        agents.append(create_agent(
            UnconventionalStrategist,
            "Unconventional Strategist",
            system_msg=f"Unconventional Strategist for: {task}\n- Generate creative strategies\n- Challenge traditional methods",
        ))

    if config.use_contrarian:
        agents.append(create_agent(
            Contrarian,
            "Contrarian",
            system_msg=f"Contrarian for: {task}\n- Challenge conventional ideas\n- Explore alternative perspectives",
        ))

    return agents

def get_initial_speaker(agents: List[Agent], config: TeamConfig) -> Optional[Agent]:
    """Determine the initial speaker based on team composition."""
    if config.use_first_principles:
        for agent in agents:
            if isinstance(agent, FirstPrinciplesAnalyst):
                return agent
    return agents[0] if agents else None

def create_initial_message(task: str, agents: List[Agent]) -> dict:
    """Create an appropriate initial message based on team composition."""
    agent_roles = "\n".join([f"- {agent.name}" for agent in agents])

    return {
        "role": "user",
        "content": (
            f"Task: {task}\n\n"
            f"Available team members:\n{agent_roles}\n\n"
            "Collaboration guidelines:\n"
            "1. Break down problem components\n"
            "2. Research relevant information\n"
            "3. Develop strategic approach\n"
            "4. Review and refine solution\n"
            "5. Consider alternative perspectives\n"
            "6. Stress-test proposals\n"
            "7. Synthesize comprehensive deliverable"
        ),
    }

def run_group_chat(task: str, config: Optional[TeamConfig] = None):
    if config is None:
        config = TeamConfig()

    agents = initialize_agents(task, config)
    if not agents:
        raise ValueError("No agents were initialized")

    group_chat = GroupChat(
        agents=agents,
        messages=[],
        max_round=config.max_rounds,
        speaker_selection_method=config.speaker_selection_method,
        allow_repeat_speaker=config.allow_repeat_speaker,
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=create_llm_config(temperature=config.temperature),
    )

    initial_message = create_initial_message(task, agents)

    try:
        initial_speaker = get_initial_speaker(agents, config)
        if not initial_speaker:
            raise ValueError("Could not determine initial speaker")

        chat_result = initial_speaker.initiate_chat(
            manager,
            message=initial_message,
            silent=False
        )

        return chat_result.chat_history if hasattr(chat_result, 'chat_history') else group_chat.messages

    except Exception as e:
        print(f"Error during chat execution: {str(e)}")
        return None

if __name__ == "__main__":
    team_config = TeamConfig(
        use_first_principles=True,
        use_devils_advocate=True,
        use_critical_friend=True,
        use_unconventional_strategist=True,
        use_contrarian=True,
        max_rounds=10,
        temperature=0.2,
        speaker_selection_method="auto",
        allow_repeat_speaker=False
    )

    task_description = """
    Discuss this topic: NVIDIA should establish a manufacturing plant in Malaysia or Indonesia?
    """

    messages = run_group_chat(task_description, team_config)
    if messages:
        print("\nChat History:")
        for msg in messages:
            if isinstance(msg, dict) and 'content' in msg:
                print(f"\n{msg.get('name', 'Unknown')}: {msg['content']}\n")
