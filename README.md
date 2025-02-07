# README: Modified Autogen Agents + Ollama Server + DeepSeek-R1 AI Model
## Ihsan Mohd Yassin, Universiti Teknologi Mara, Malaysia

## Overview

This project implements a multi-agent conversational framework designed to tackle complex problems by leveraging specialized agents with distinct check roles. Each agent operates autonomously and collaboratively, ensuring a comprehensive analysis and synthesis of solutions. Please install Ollama for PC: https://ollama.com/download and pull the deepseek-r1 AI model into your computer. 

I think this project has significant potential since we can implement this agentic framework entirely in our own computer.

Do help me improve this further.

Cheers!

### Features:

1. Modular agent architecture for flexible team composition.
2. Agents specializing in different analytical and strategic perspectives:
   - **First Principles Analyst**: Breaks down problems to their core components.
   - **Devil's Advocate**: Identifies flaws and explores failure scenarios.
   - **Critical Friend**: Provides constructive feedback and supportive critique.
   - **Unconventional Strategist**: Generates creative strategies and challenges traditional methods.
   - **Contrarian**: Challenges assumptions and explores alternative perspectives.
3. Configurable team setup via the `TeamConfig` class.
4. Centralized communication using the `GroupChat` and `GroupChatManager` components.
5. Each agent talks to each other, searches the internet and does LLM research (through its own "personal" ChatGPT powered by DeepSeek-R1)

## Components

### 1. **Agents**

Each agent specializes in a specific role and contributes uniquely to problem-solving:

- **First Principles Analyst**:

  - Foundational analysis and assumption validation.
  - Breaks down problems into first principles.

- **Devil's Advocate**:

  - Focuses on flaw detection and failure scenario exploration.
  - Challenges consensus to identify risks.

- **Critical Friend**:

  - Offers constructive feedback and identifies areas for improvement.
  - Ensures alignment with objectives and goals.

- **Unconventional Strategist**:

  - Proposes innovative strategies and disrupts traditional methods.
  - Evaluates scalability and feasibility of creative solutions.

- **Contrarian**:

  - Challenges conventional thinking and proposes alternative perspectives.
  - Identifies hidden opportunities and weaknesses.

### 2. **Group Chat**

- Centralized communication system for agents to exchange information.
- Managed by `GroupChatManager`, which oversees message flow and coordination.

### 3. **LLM Integration**

- Agents utilize LLMs for advanced reasoning, analysis, and decision-making.
- Configurable through `create_llm_config` for temperature and model settings.

## Configuration

### TeamConfig

The `TeamConfig` class allows for flexible team composition and behavior tuning:

```python
@dataclass
class TeamConfig:
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
```

### Example Configuration:

```python
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
```

## Workflow

1. Define the task or problem for discussion.
2. Configure the team using `TeamConfig`.
3. Initialize agents and start the conversation via `run_group_chat()`.
4. Agents collaborate autonomously to analyze the problem and synthesize solutions.
5. The final output is a comprehensive discussion log, including insights from all agents.

### Example Task:

```python
task_description = """
Discuss this topic: NVIDIA should establish a manufacturing plant in Malaysia or Indonesia?
"""

messages = run_group_chat(task_description, team_config)
if messages:
    for msg in messages:
        print(f"{msg.get('name', 'Unknown')}: {msg['content']}")
```

## Dependencies & Execution
Copy the entire project to your computer, then run from autogen_bolt.
It depends on some original autogen packages, so the package (along with many others) should be installed as well.

## License

This project is licensed under the MIT License.

## Contact

For questions or contributions, please reach out to ihsan.yassin@gmail.com. 
https://www.ihsanyassin.com
Do contact for collaborations!
Thank you.

