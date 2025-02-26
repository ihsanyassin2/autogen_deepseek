# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import copy
import json
import logging
import random
import re
import sys
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, Any

from autogen.code_utils import content_str
from autogen.exception_utils import AgentNameConflict, NoEligibleSpeaker, UndefinedNextAgent
from autogen.formatting_utils import colored
from autogen.graph_utils import check_graph_validity, invert_disallowed_to_allowed
from autogen.io.base import IOStream
from autogen.oai.client import ModelClient
from autogen.runtime_logging import log_new_agent, logging_enabled
from autogen import Agent
from autogen.agentchat.contrib.capabilities import transform_messages

from custom_conversable_agent import ConversableAgent

logger = logging.getLogger(__name__)

@dataclass
class GroupChat:
    """(In preview) A group chat class that contains the following data fields:
    - agents: a list of participating agents.
    - messages: a list of messages in the group chat.
    - max_round: the maximum number of rounds.
    - admin_name: the name of the admin agent if there is one. Default is "Admin".
        KeyBoardInterrupt will make the admin agent take over.
    - func_call_filter: whether to enforce function call filter. Default is True.
        When set to True and when a message is a function call suggestion,
        the next speaker will be chosen from an agent which contains the corresponding function name
        in its `function_map`.
    - select_speaker_message_template: customize the select speaker message (used in "auto" speaker selection), which appears first in the message context and generally includes the agent descriptions and list of agents. If the string contains "{roles}" it will replaced with the agent's and their role descriptions. If the string contains "{agentlist}" it will be replaced with a comma-separated list of agent names in square brackets. The default value is:
        "You are in a role play game. The following roles are available:
                {roles}.
                Read the following conversation.
                Then select the next role from {agentlist} to play. Only return the role."
    - select_speaker_prompt_template: customize the select speaker prompt (used in "auto" speaker selection), which appears last in the message context and generally includes the list of agents and guidance for the LLM to select the next agent. If the string contains "{agentlist}" it will be replaced with a comma-separated list of agent names in square brackets. The default value is:
        "Read the above conversation. Then select the next role from {agentlist} to play. Only return the role."
        To ignore this prompt being used, set this to None. If set to None, ensure your instructions for selecting a speaker are in the select_speaker_message_template string.
    - select_speaker_auto_multiple_template: customize the follow-up prompt used when selecting a speaker fails with a response that contains multiple agent names. This prompt guides the LLM to return just one agent name. Applies only to "auto" speaker selection method. If the string contains "{agentlist}" it will be replaced with a comma-separated list of agent names in square brackets. The default value is:
        "You provided more than one name in your text, please return just the name of the next speaker. To determine the speaker use these prioritised rules:
                1. If the context refers to themselves as a speaker e.g. "As the..." , choose that speaker's name
                2. If it refers to the "next" speaker name, choose that name
                3. Otherwise, choose the first provided speaker's name in the context
                The names are case-sensitive and should not be abbreviated or changed.
                Respond with ONLY the name of the speaker and DO NOT provide a reason."
    - select_speaker_auto_none_template: customize the follow-up prompt used when selecting a speaker fails with a response that contains no agent names. This prompt guides the LLM to return an agent name and provides a list of agent names. Applies only to "auto" speaker selection method. If the string contains "{agentlist}" it will be replaced with a comma-separated list of agent names in square brackets. The default value is:
        "You didn't choose a speaker. As a reminder, to determine the speaker use these prioritised rules:
                1. If the context refers to themselves as a speaker e.g. "As the..." , choose that speaker's name
                2. If it refers to the "next" speaker name, choose that name
                3. Otherwise, choose the first provided speaker's name in the context
                The names are case-sensitive and should not be abbreviated or changed.
                The only names that are accepted are {agentlist}.
                Respond with ONLY the name of the speaker and DO NOT provide a reason."
    - speaker_selection_method: the method for selecting the next speaker. Default is "auto".
        Could be any of the following (case insensitive), will raise ValueError if not recognized:
        - "auto": the next speaker is selected automatically by LLM.
        - "manual": the next speaker is selected manually by user input.
        - "random": the next speaker is selected randomly.
        - "round_robin": the next speaker is selected in a round robin fashion, i.e., iterating in the same order as provided in `agents`.
        - a customized speaker selection function (Callable): the function will be called to select the next speaker.
            The function should take the last speaker and the group chat as input and return one of the following:
                1. an `Agent` class, it must be one of the agents in the group chat.
                2. a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
                3. None, which would terminate the conversation gracefully.
            ```python
            def custom_speaker_selection_func(
                last_speaker: Agent, groupchat: GroupChat
            ) -> Union[Agent, str, None]:
            ```
    - max_retries_for_selecting_speaker: the maximum number of times the speaker selection requery process will run.
        If, during speaker selection, multiple agent names or no agent names are returned by the LLM as the next agent, it will be queried again up to the maximum number
        of times until a single agent is returned or it exhausts the maximum attempts.
        Applies only to "auto" speaker selection method.
        Default is 2.
    - select_speaker_transform_messages: (optional) the message transformations to apply to the nested select speaker agent-to-agent chat messages.
        Takes a TransformMessages object, defaults to None and is only utilised when the speaker selection method is "auto".
    - select_speaker_auto_verbose: whether to output the select speaker responses and selections
        If set to True, the outputs from the two agents in the nested select speaker chat will be output, along with
        whether the responses were successful, or not, in selecting an agent
        Applies only to "auto" speaker selection method.
    - allow_repeat_speaker: whether to allow the same speaker to speak consecutively.
        Default is True, in which case all speakers are allowed to speak consecutively.
        If `allow_repeat_speaker` is a list of Agents, then only those listed agents are allowed to repeat.
        If set to False, then no speakers are allowed to repeat.
        `allow_repeat_speaker` and `allowed_or_disallowed_speaker_transitions` are mutually exclusive.
    - allowed_or_disallowed_speaker_transitions: dict.
        The keys are source agents, and the values are agents that the key agent can/can't transit to,
        depending on speaker_transitions_type. Default is None, which means all agents can transit to all other agents.
        `allow_repeat_speaker` and `allowed_or_disallowed_speaker_transitions` are mutually exclusive.
    - speaker_transitions_type: whether the speaker_transitions_type is a dictionary containing lists of allowed agents or disallowed agents.
        "allowed" means the `allowed_or_disallowed_speaker_transitions` is a dictionary containing lists of allowed agents.
        If set to "disallowed", then the `allowed_or_disallowed_speaker_transitions` is a dictionary containing lists of disallowed agents.
        Must be supplied if `allowed_or_disallowed_speaker_transitions` is not None.
    - enable_clear_history: enable possibility to clear history of messages for agents manually by providing
        "clear history" phrase in user prompt. This is experimental feature.
        See description of GroupChatManager.clear_agents_history function for more info.
    - send_introductions: send a round of introductions at the start of the group chat, so agents know who they can speak to (default: False)
    - select_speaker_auto_model_client_cls: Custom model client class for the internal speaker select agent used during 'auto' speaker selection (optional)
    - select_speaker_auto_llm_config: LLM config for the internal speaker select agent used during 'auto' speaker selection (optional)
    - role_for_select_speaker_messages: sets the role name for speaker selection when in 'auto' mode, typically 'user' or 'system'. (default: 'system')
    """

    agents: list[Agent]
    messages: list[dict]
    max_round: int = 10
    admin_name: str = "Admin"
    func_call_filter: bool = True
    speaker_selection_method: Union[Literal["auto", "manual", "random", "round_robin"], Callable] = "auto"
    max_retries_for_selecting_speaker: int = 2
    allow_repeat_speaker: Optional[Union[bool, list[Agent]]] = None
    allowed_or_disallowed_speaker_transitions: Optional[dict] = None
    speaker_transitions_type: Literal["allowed", "disallowed", None] = None
    enable_clear_history: bool = False
    send_introductions: bool = False
    select_speaker_message_template: str = """You are in a role play game. The following roles are available:
                {roles}.
                Read the following conversation.
                Then select the next role from {agentlist} to play. Only return the role."""
    select_speaker_prompt_template: str = (
        "Read the above conversation. Then select the next role from {agentlist} to play. Only return the role."
    )
    select_speaker_auto_multiple_template: str = """You provided more than one name in your text, please return just the name of the next speaker. To determine the speaker use these prioritised rules:
    1. If the context refers to themselves as a speaker e.g. "As the..." , choose that speaker's name
    2. If it refers to the "next" speaker name, choose that name
    3. Otherwise, choose the first provided speaker's name in the context
    The names are case-sensitive and should not be abbreviated or changed.
    Respond with ONLY the name of the speaker and DO NOT provide a reason."""
    select_speaker_auto_none_template: str = """You didn't choose a speaker. As a reminder, to determine the speaker use these prioritised rules:
    1. If the context refers to themselves as a speaker e.g. "As the..." , choose that speaker's name
    2. If it refers to the "next" speaker name, choose that name
    3. Otherwise, choose the first provided speaker's name in the context
    The names are case-sensitive and should not be abbreviated or changed.
    The only names that are accepted are {agentlist}.
    Respond with ONLY the name of the speaker and DO NOT provide a reason."""
    select_speaker_transform_messages: Optional[transform_messages.TransformMessages] = None
    select_speaker_auto_verbose: Optional[bool] = False
    select_speaker_auto_model_client_cls: Optional[Union[ModelClient, list[ModelClient]]] = None
    select_speaker_auto_llm_config: Optional[Union[dict, Literal[False]]] = None
    role_for_select_speaker_messages: Optional[str] = "system"

    _VALID_SPEAKER_SELECTION_METHODS = ["auto", "manual", "random", "round_robin"]
    _VALID_SPEAKER_TRANSITIONS_TYPE = ["allowed", "disallowed", None]

    # Define a class attribute for the default introduction message
    DEFAULT_INTRO_MSG = (
        "Hello everyone. We have assembled a great team today to answer questions and solve tasks. In attendance are:"
    )

    allowed_speaker_transitions_dict: dict = field(init=False)

    def __post_init__(self):
        # Post init steers clears of the automatically generated __init__ method from dataclass

        if self.allow_repeat_speaker is not None and not isinstance(self.allow_repeat_speaker, (bool, list)):
            raise ValueError("GroupChat allow_repeat_speaker should be a bool or a list of Agents.")

        # Here, we create allowed_speaker_transitions_dict from the supplied allowed_or_disallowed_speaker_transitions and speaker_transitions_type, and lastly checks for validity.

        # Check input
        if self.speaker_transitions_type is not None:
            self.speaker_transitions_type = self.speaker_transitions_type.lower()

        if self.speaker_transitions_type not in self._VALID_SPEAKER_TRANSITIONS_TYPE:
            raise ValueError(
                f"GroupChat speaker_transitions_type is set to '{self.speaker_transitions_type}'. "
                f"It should be one of {self._VALID_SPEAKER_TRANSITIONS_TYPE} (case insensitive). "
            )

        # If both self.allowed_or_disallowed_speaker_transitions is None and self.allow_repeat_speaker is None, set allow_repeat_speaker to True to ensure backward compatibility
        # Discussed in https://github.com/microsoft/autogen/pull/857#discussion_r1451541204
        if self.allowed_or_disallowed_speaker_transitions is None and self.allow_repeat_speaker is None:
            self.allow_repeat_speaker = True

        # self.allowed_or_disallowed_speaker_transitions and self.allow_repeat_speaker are mutually exclusive parameters.
        # Discussed in https://github.com/microsoft/autogen/pull/857#discussion_r1451266661
        if self.allowed_or_disallowed_speaker_transitions is not None and self.allow_repeat_speaker is not None:
            raise ValueError(
                "Don't provide both allowed_or_disallowed_speaker_transitions and allow_repeat_speaker in group chat. "
                "Please set one of them to None."
            )

        # Asks the user to specify whether the speaker_transitions_type is allowed or disallowed if speaker_transitions_type is supplied
        # Discussed in https://github.com/microsoft/autogen/pull/857#discussion_r1451259524
        if self.allowed_or_disallowed_speaker_transitions is not None and self.speaker_transitions_type is None:
            raise ValueError(
                "GroupChat allowed_or_disallowed_speaker_transitions is not None, but speaker_transitions_type is None. "
                "Please set speaker_transitions_type to either 'allowed' or 'disallowed'."
            )

        # Inferring self.allowed_speaker_transitions_dict
        # Create self.allowed_speaker_transitions_dict if allowed_or_disallowed_speaker_transitions is None, using allow_repeat_speaker
        if self.allowed_or_disallowed_speaker_transitions is None:
            self.allowed_speaker_transitions_dict = {}

            # Create a fully connected allowed_speaker_transitions_dict not including self loops
            for agent in self.agents:
                self.allowed_speaker_transitions_dict[agent] = [
                    other_agent for other_agent in self.agents if other_agent != agent
                ]

            # If self.allow_repeat_speaker is True, add self loops to all agents
            if self.allow_repeat_speaker is True:
                for agent in self.agents:
                    self.allowed_speaker_transitions_dict[agent].append(agent)

            # Else if self.allow_repeat_speaker is a list of Agents, add self loops to the agents in the list
            elif isinstance(self.allow_repeat_speaker, list):
                for agent in self.allow_repeat_speaker:
                    self.allowed_speaker_transitions_dict[agent].append(agent)

        # Create self.allowed_speaker_transitions_dict if allowed_or_disallowed_speaker_transitions is not None, using allowed_or_disallowed_speaker_transitions
        else:
            # Process based on speaker_transitions_type
            if self.speaker_transitions_type == "allowed":
                self.allowed_speaker_transitions_dict = self.allowed_or_disallowed_speaker_transitions
            else:
                # Logic for processing disallowed allowed_or_disallowed_speaker_transitions to allowed_speaker_transitions_dict
                self.allowed_speaker_transitions_dict = invert_disallowed_to_allowed(
                    self.allowed_or_disallowed_speaker_transitions, self.agents
                )

        # Check for validity
        check_graph_validity(
            allowed_speaker_transitions_dict=self.allowed_speaker_transitions_dict,
            agents=self.agents,
        )

        # Check select speaker messages, prompts, roles, and retries have values
        if self.select_speaker_message_template is None or len(self.select_speaker_message_template) == 0:
            raise ValueError("select_speaker_message_template cannot be empty or None.")

        if self.select_speaker_prompt_template is not None and len(self.select_speaker_prompt_template) == 0:
            self.select_speaker_prompt_template = None

        if self.role_for_select_speaker_messages is None or len(self.role_for_select_speaker_messages) == 0:
            raise ValueError("role_for_select_speaker_messages cannot be empty or None.")

        if self.select_speaker_auto_multiple_template is None or len(self.select_speaker_auto_multiple_template) == 0:
            raise ValueError("select_speaker_auto_multiple_template cannot be empty or None.")

        if self.select_speaker_auto_none_template is None or len(self.select_speaker_auto_none_template) == 0:
            raise ValueError("select_speaker_auto_none_template cannot be empty or None.")

        if self.max_retries_for_selecting_speaker is None or len(self.role_for_select_speaker_messages) == 0:
            raise ValueError("role_for_select_speaker_messages cannot be empty or None.")

        # Validate max select speakers retries
        if self.max_retries_for_selecting_speaker is None or not isinstance(
            self.max_retries_for_selecting_speaker, int
        ):
            raise ValueError("max_retries_for_selecting_speaker cannot be None or non-int")
        elif self.max_retries_for_selecting_speaker < 0:
            raise ValueError("max_retries_for_selecting_speaker must be greater than or equal to zero")

        # Load message transforms here (load once for the Group Chat so we don't have to re-initiate it and it maintains the cache across subsequent select speaker calls)
        if self.select_speaker_transform_messages is not None:
            if isinstance(self.select_speaker_transform_messages, transform_messages.TransformMessages):
                self._speaker_selection_transforms = self.select_speaker_transform_messages
            else:
                raise ValueError("select_speaker_transform_messages must be None or MessageTransforms.")
        else:
            self._speaker_selection_transforms = None

        # Validate select_speaker_auto_verbose
        if self.select_speaker_auto_verbose is None or not isinstance(self.select_speaker_auto_verbose, bool):
            raise ValueError("select_speaker_auto_verbose cannot be None or non-bool")

    @property
    def agent_names(self) -> list[str]:
        """Return the names of the agents in the group chat."""
        return [agent.name for agent in self.agents]

    def reset(self):
        """Reset the group chat."""
        self.messages.clear()

    def append(self, message: dict, speaker: Agent):
        """Append a message to the group chat.
        We cast the content to str here so that it can be managed by text-based
        model.
        """
        # set the name to speaker's name if the role is not function
        # if the role is tool, it is OK to modify the name
        if message["role"] != "function":
            message["name"] = speaker.name
        if not isinstance(message["content"], str) and not isinstance(message["content"], list):
            message["content"] = str(message["content"])
        message["content"] = content_str(message["content"])
        self.messages.append(message)

    def agent_by_name(
        self, name: str, recursive: bool = False, raise_on_name_conflict: bool = False
    ) -> Optional[Agent]:
        """Returns the agent with a given name. If recursive is True, it will search in nested teams."""
        agents = self.nested_agents() if recursive else self.agents
        filtered_agents = [agent for agent in agents if agent.name == name]

        if raise_on_name_conflict and len(filtered_agents) > 1:
            raise AgentNameConflict()

        return filtered_agents[0] if filtered_agents else None

    def nested_agents(self) -> list[Agent]:
        """Returns all agents in the group chat manager."""
        agents = self.agents.copy()
        for agent in agents:
            if isinstance(agent, GroupChatManager):
                # Recursive call for nested teams
                agents.extend(agent.groupchat.nested_agents())
        return agents

    def next_agent(self, agent: Agent, agents: Optional[list[Agent]] = None) -> Agent:
        """Return the next agent in the list."""
        if agents is None:
            agents = self.agents

        # Ensure the provided list of agents is a subset of self.agents
        if not set(agents).issubset(set(self.agents)):
            raise UndefinedNextAgent()

        # What index is the agent? (-1 if not present)
        idx = self.agent_names.index(agent.name) if agent.name in self.agent_names else -1

        # Return the next agent
        if agents == self.agents:
            return agents[(idx + 1) % len(agents)]
        else:
            offset = idx + 1
            for i in range(len(self.agents)):
                if self.agents[(offset + i) % len(self.agents)] in agents:
                    return self.agents[(offset + i) % len(self.agents)]

        # Explicitly handle cases where no valid next agent exists in the provided subset.
        raise UndefinedNextAgent()

    def select_speaker_msg(self, agents: Optional[list[Agent]] = None) -> str:
        """Return the system message for selecting the next speaker. This is always the *first* message in the context."""
        if agents is None:
            agents = self.agents

        roles = self._participant_roles(agents)
        agentlist = f"{[agent.name for agent in agents]}"

        return_msg = self.select_speaker_message_template.format(roles=roles, agentlist=agentlist)
        return return_msg

    def select_speaker_prompt(self, agents: Optional[list[Agent]] = None) -> str:
        """Return the floating system prompt selecting the next speaker.
        This is always the *last* message in the context.
        Will return None if the select_speaker_prompt_template is None."""

        if self.select_speaker_prompt_template is None:
            return None

        if agents is None:
            agents = self.agents

        agentlist = f"{[agent.name for agent in agents]}"

        return_prompt = self.select_speaker_prompt_template.format(agentlist=agentlist)
        return return_prompt

    def introductions_msg(self, agents: Optional[list[Agent]] = None) -> str:
        """Return the system message for selecting the next speaker. This is always the *first* message in the context."""
        if agents is None:
            agents = self.agents

        # Use the class attribute instead of a hardcoded string
        intro_msg = self.DEFAULT_INTRO_MSG
        participant_roles = self._participant_roles(agents)

        return f"{intro_msg}\n\n{participant_roles}"

    def manual_select_speaker(self, agents: Optional[list[Agent]] = None) -> Union[Agent, None]:
        """Manually select the next speaker."""
        iostream = IOStream.get_default()

        if agents is None:
            agents = self.agents

        iostream.print("Please select the next speaker from the following list:")
        _n_agents = len(agents)
        for i in range(_n_agents):
            iostream.print(f"{i+1}: {agents[i].name}")
        try_count = 0
        # Assume the user will enter a valid number within 3 tries, otherwise use auto selection to avoid blocking.
        while try_count <= 3:
            try_count += 1
            if try_count >= 3:
                iostream.print(f"You have tried {try_count} times. The next speaker will be selected automatically.")
                break
            try:
                i = iostream.input(
                    "Enter the number of the next speaker (enter nothing or `q` to use auto selection): "
                )
                if i == "" or i == "q":
                    break
                i = int(i)
                if i > 0 and i <= _n_agents:
                    return agents[i - 1]
                else:
                    raise ValueError
            except ValueError:
                iostream.print(f"Invalid input. Please enter a number between 1 and {_n_agents}.")
        return None

    def random_select_speaker(self, agents: Optional[list[Agent]] = None) -> Union[Agent, None]:
        """Randomly select the next speaker."""
        if agents is None:
            agents = self.agents
        return random.choice(agents)

    def _prepare_and_select_agents(
        self,
        last_speaker: Agent,
    ) -> tuple[Optional[Agent], list[Agent], Optional[list[dict]]]:
        # If self.speaker_selection_method is a callable, call it to get the next speaker.
        # If self.speaker_selection_method is a string, return it.
        speaker_selection_method = self.speaker_selection_method
        if isinstance(self.speaker_selection_method, Callable):
            selected_agent = self.speaker_selection_method(last_speaker, self)
            if selected_agent is None:
                raise NoEligibleSpeaker("Custom speaker selection function returned None. Terminating conversation.")
            elif isinstance(selected_agent, Agent):
                if selected_agent in self.agents:
                    return selected_agent, self.agents, None
                else:
                    raise ValueError(
                        f"Custom speaker selection function returned an agent {selected_agent.name} not in the group chat."
                    )
            elif isinstance(selected_agent, str):
                # If returned a string, assume it is a speaker selection method
                speaker_selection_method = selected_agent
            else:
                raise ValueError(
                    f"Custom speaker selection function returned an object of type {type(selected_agent)} instead of Agent or str."
                )

        if speaker_selection_method.lower() not in self._VALID_SPEAKER_SELECTION_METHODS:
            raise ValueError(
                f"GroupChat speaker_selection_method is set to '{speaker_selection_method}'. "
                f"It should be one of {self._VALID_SPEAKER_SELECTION_METHODS} (case insensitive). "
            )

        # If provided a list, make sure the agent is in the list
        allow_repeat_speaker = (
            self.allow_repeat_speaker
            if isinstance(self.allow_repeat_speaker, bool) or self.allow_repeat_speaker is None
            else last_speaker in self.allow_repeat_speaker
        )

        agents = self.agents
        n_agents = len(agents)
        # Warn if GroupChat is underpopulated
        if n_agents < 2:
            raise ValueError(
                f"GroupChat is underpopulated with {n_agents} agents. "
                "Please add more agents to the GroupChat or use direct communication instead."
            )
        elif n_agents == 2 and speaker_selection_method.lower() != "round_robin" and allow_repeat_speaker:
            logger.warning(
                f"GroupChat is underpopulated with {n_agents} agents. "
                "Consider setting speaker_selection_method to 'round_robin' or allow_repeat_speaker to False, "
                "or use direct communication, unless repeated speaker is desired."
            )

        if (
            self.func_call_filter
            and self.messages
            and ("function_call" in self.messages[-1] or "tool_calls" in self.messages[-1])
        ):
            funcs = []
            if "function_call" in self.messages[-1]:
                funcs += [self.messages[-1]["function_call"]["name"]]
            if "tool_calls" in self.messages[-1]:
                funcs += [
                    tool["function"]["name"] for tool in self.messages[-1]["tool_calls"] if tool["type"] == "function"
                ]

            # find agents with the right function_map which contains the function name
            agents = [agent for agent in self.agents if agent.can_execute_function(funcs)]
            if len(agents) == 1:
                # only one agent can execute the function
                return agents[0], agents, None
            elif not agents:
                # find all the agents with function_map
                agents = [agent for agent in self.agents if agent.function_map]
                if len(agents) == 1:
                    return agents[0], agents, None
                elif not agents:
                    raise ValueError(
                        f"No agent can execute the function {', '.join(funcs)}. "
                        "Please check the function_map of the agents."
                    )
        # remove the last speaker from the list to avoid selecting the same speaker if allow_repeat_speaker is False
        agents = [agent for agent in agents if agent != last_speaker] if allow_repeat_speaker is False else agents

        # Filter agents with allowed_speaker_transitions_dict

        is_last_speaker_in_group = last_speaker in self.agents

        # this condition means last_speaker is a sink in the graph, then no agents are eligible
        if last_speaker not in self.allowed_speaker_transitions_dict and is_last_speaker_in_group:
            raise NoEligibleSpeaker(f"Last speaker {last_speaker.name} is not in the allowed_speaker_transitions_dict.")
        # last_speaker is not in the group, so all agents are eligible
        elif last_speaker not in self.allowed_speaker_transitions_dict and not is_last_speaker_in_group:
            graph_eligible_agents = []
        else:
            # Extract agent names from the list of agents
            graph_eligible_agents = [
                agent for agent in agents if agent in self.allowed_speaker_transitions_dict[last_speaker]
            ]

        # If there is only one eligible agent, just return it to avoid the speaker selection prompt
        if len(graph_eligible_agents) == 1:
            return graph_eligible_agents[0], graph_eligible_agents, None

        # If there are no eligible agents, return None, which means all agents will be taken into consideration in the next step
        if len(graph_eligible_agents) == 0:
            graph_eligible_agents = None

        # Use the selected speaker selection method
        select_speaker_messages = None
        if speaker_selection_method.lower() == "manual":
            selected_agent = self.manual_select_speaker(graph_eligible_agents)
        elif speaker_selection_method.lower() == "round_robin":
            selected_agent = self.next_agent(last_speaker, graph_eligible_agents)
        elif speaker_selection_method.lower() == "random":
            selected_agent = self.random_select_speaker(graph_eligible_agents)
        else:  # auto
            selected_agent = None
            select_speaker_messages = self.messages.copy()
            # If last message is a tool call or function call, blank the call so the api doesn't throw
            if select_speaker_messages[-1].get("function_call", False):
                select_speaker_messages[-1] = dict(select_speaker_messages[-1], function_call=None)
            if select_speaker_messages[-1].get("tool_calls", False):
                select_speaker_messages[-1] = dict(select_speaker_messages[-1], tool_calls=None)
        return selected_agent, graph_eligible_agents, select_speaker_messages

    def select_speaker(self, last_speaker: Agent, selector: ConversableAgent) -> Agent:
        """Select the next speaker (with requery)."""

        # Prepare the list of available agents and select an agent if selection method allows (non-auto)
        selected_agent, agents, messages = self._prepare_and_select_agents(last_speaker)
        if selected_agent:
            return selected_agent
        elif self.speaker_selection_method == "manual":
            # An agent has not been selected while in manual mode, so move to the next agent
            return self.next_agent(last_speaker)

        # auto speaker selection with 2-agent chat
        return self._auto_select_speaker(last_speaker, selector, messages, agents)

    async def a_select_speaker(self, last_speaker: Agent, selector: ConversableAgent) -> Agent:
        """Select the next speaker (with requery), asynchronously."""

        selected_agent, agents, messages = self._prepare_and_select_agents(last_speaker)
        if selected_agent:
            return selected_agent
        elif self.speaker_selection_method == "manual":
            # An agent has not been selected while in manual mode, so move to the next agent
            return self.next_agent(last_speaker)

        # auto speaker selection with 2-agent chat
        return await self.a_auto_select_speaker(last_speaker, selector, messages, agents)

    def _finalize_speaker(self, last_speaker: Agent, final: bool, name: str, agents: Optional[list[Agent]]) -> Agent:
        if not final:
            # the LLM client is None, thus no reply is generated. Use round robin instead.
            return self.next_agent(last_speaker, agents)

        # If exactly one agent is mentioned, use it. Otherwise, leave the OAI response unmodified
        mentions = self._mentioned_agents(name, agents)
        if len(mentions) == 1:
            name = next(iter(mentions))
        else:
            logger.warning(
                f"GroupChat select_speaker failed to resolve the next speaker's name. This is because the speaker selection OAI call returned:\n{name}"
            )

        # Return the result
        agent = self.agent_by_name(name)
        return agent if agent else self.next_agent(last_speaker, agents)

    def _register_client_from_config(self, agent: Agent, config: dict):
        model_client_cls_to_match = config.get("model_client_cls")
        if model_client_cls_to_match:
            if not self.select_speaker_auto_model_client_cls:
                raise ValueError(
                    "A custom model was detected in the config but no 'model_client_cls' "
                    "was supplied for registration in GroupChat."
                )

            if isinstance(self.select_speaker_auto_model_client_cls, list):
                # Register the first custom model client class matching the name specified in the config
                matching_model_cls = [
                    client_cls
                    for client_cls in self.select_speaker_auto_model_client_cls
                    if client_cls.__name__ == model_client_cls_to_match
                ]
                if len(set(matching_model_cls)) > 1:
                    raise RuntimeError(
                        f"More than one unique 'model_client_cls' with __name__ '{model_client_cls_to_match}'."
                    )
                if not matching_model_cls:
                    raise ValueError(
                        "No model's __name__ matches the model client class "
                        f"'{model_client_cls_to_match}' specified in select_speaker_auto_llm_config."
                    )
                select_speaker_auto_model_client_cls = matching_model_cls[0]
            else:
                # Register the only custom model client
                select_speaker_auto_model_client_cls = self.select_speaker_auto_model_client_cls

            agent.register_model_client(select_speaker_auto_model_client_cls)

    def _register_custom_model_clients(self, agent: ConversableAgent):
        if not self.select_speaker_auto_llm_config:
            return

        config_format_is_list = "config_list" in self.select_speaker_auto_llm_config.keys()
        if config_format_is_list:
            for config in self.select_speaker_auto_llm_config["config_list"]:
                self._register_client_from_config(agent, config)
        elif not config_format_is_list:
            self._register_client_from_config(agent, self.select_speaker_auto_llm_config)

    def _create_internal_agents(
        self, agents, max_attempts, messages, validate_speaker_name, selector: Optional[ConversableAgent] = None
    ):
        checking_agent = ConversableAgent("checking_agent", default_auto_reply=max_attempts)

        # Register the speaker validation function with the checking agent
        checking_agent.register_reply(
            [ConversableAgent, None],
            reply_func=validate_speaker_name,  # Validate each response
            remove_other_reply_funcs=True,
        )

        # Override the selector's config if one was passed as a parameter to this class
        speaker_selection_llm_config = self.select_speaker_auto_llm_config or selector.llm_config

        # Agent for selecting a single agent name from the response
        speaker_selection_agent = ConversableAgent(
            "speaker_selection_agent",
            system_message=self.select_speaker_msg(agents),
            chat_messages={checking_agent: messages},
            llm_config=speaker_selection_llm_config,
            human_input_mode="NEVER",
            # Suppresses some extra terminal outputs, outputs will be handled by select_speaker_auto_verbose
        )

        # Register any custom model passed in select_speaker_auto_llm_config with the speaker_selection_agent
        self._register_custom_model_clients(speaker_selection_agent)

        return checking_agent, speaker_selection_agent

    def _auto_select_speaker(
        self,
        last_speaker: Agent,
        selector: ConversableAgent,
        messages: Optional[list[dict]],
        agents: Optional[list[Agent]],
    ) -> Agent:
        """Selects next speaker for the "auto" speaker selection method. Utilises its own two-agent chat to determine the next speaker and supports requerying.

        Speaker selection for "auto" speaker selection method:
        1. Create a two-agent chat with a speaker selector agent and a speaker validator agent, like a nested chat
        2. Inject the group messages into the new chat
        3. Run the two-agent chat, evaluating the result of response from the speaker selector agent:
            - If a single agent is provided then we return it and finish. If not, we add an additional message to this nested chat in an attempt to guide the LLM to a single agent response
        4. Chat continues until a single agent is nominated or there are no more attempts left
        5. If we run out of turns and no single agent can be determined, the next speaker in the list of agents is returned

        Args:
            last_speaker Agent: The previous speaker in the group chat
            selector ConversableAgent:
            messages Optional[List[Dict]]: Current chat messages
            agents Optional[List[Agent]]: Valid list of agents for speaker selection

        Returns:
            Dict: a counter for mentioned agents.
        """

        # If no agents are passed in, assign all the group chat's agents
        if agents is None:
            agents = self.agents

        # The maximum number of speaker selection attempts (including requeries)
        # is the initial speaker selection attempt plus the maximum number of retries.
        # We track these and use them in the validation function as we can't
        # access the max_turns from within validate_speaker_name.
        max_attempts = 1 + self.max_retries_for_selecting_speaker
        attempts_left = max_attempts
        attempt = 0

        # Registered reply function for checking_agent, checks the result of the response for agent names
        def validate_speaker_name(recipient, messages, sender, config) -> tuple[bool, Union[str, dict, None]]:
            # The number of retries left, starting at max_retries_for_selecting_speaker
            nonlocal attempts_left
            nonlocal attempt

            attempt = attempt + 1
            attempts_left = attempts_left - 1

            return self._validate_speaker_name(recipient, messages, sender, config, attempts_left, attempt, agents)

        # Two-agent chat for speaker selection

        # Agent for checking the response from the speaker_select_agent
        checking_agent, speaker_selection_agent = self._create_internal_agents(
            agents, max_attempts, messages, validate_speaker_name, selector
        )

        # Create the starting message
        if self.select_speaker_prompt_template is not None:
            start_message = {
                "content": self.select_speaker_prompt(agents),
                "name": "checking_agent",
                "override_role": self.role_for_select_speaker_messages,
            }
        else:
            start_message = messages[-1]

        # Add the message transforms, if any, to the speaker selection agent
        if self._speaker_selection_transforms is not None:
            self._speaker_selection_transforms.add_to_agent(speaker_selection_agent)

        # Run the speaker selection chat
        result = checking_agent.initiate_chat(
            speaker_selection_agent,
            cache=None,  # don't use caching for the speaker selection chat
            message=start_message,
            max_turns=2
            * max(1, max_attempts),  # Limiting the chat to the number of attempts, including the initial one
            clear_history=False,
            silent=not self.select_speaker_auto_verbose,  # Base silence on the verbose attribute
        )

        return self._process_speaker_selection_result(result, last_speaker, agents)

    async def a_auto_select_speaker(
        self,
        last_speaker: Agent,
        selector: ConversableAgent,
        messages: Optional[list[dict]],
        agents: Optional[list[Agent]],
    ) -> Agent:
        """(Asynchronous) Selects next speaker for the "auto" speaker selection method. Utilises its own two-agent chat to determine the next speaker and supports requerying.

        Speaker selection for "auto" speaker selection method:
        1. Create a two-agent chat with a speaker selector agent and a speaker validator agent, like a nested chat
        2. Inject the group messages into the new chat
        3. Run the two-agent chat, evaluating the result of response from the speaker selector agent:
            - If a single agent is provided then we return it and finish. If not, we add an additional message to this nested chat in an attempt to guide the LLM to a single agent response
        4. Chat continues until a single agent is nominated or there are no more attempts left
        5. If we run out of turns and no single agent can be determined, the next speaker in the list of agents is returned

        Args:
            last_speaker Agent: The previous speaker in the group chat
            selector ConversableAgent:
            messages Optional[List[Dict]]: Current chat messages
            agents Optional[List[Agent]]: Valid list of agents for speaker selection

        Returns:
            Dict: a counter for mentioned agents.
        """

        # If no agents are passed in, assign all the group chat's agents
        if agents is None:
            agents = self.agents

        # The maximum number of speaker selection attempts (including requeries)
        # We track these and use them in the validation function as we can't
        # access the max_turns from within validate_speaker_name
        max_attempts = 1 + self.max_retries_for_selecting_speaker
        attempts_left = max_attempts
        attempt = 0

        # Registered reply function for checking_agent, checks the result of the response for agent names
        def validate_speaker_name(recipient, messages, sender, config) -> tuple[bool, Union[str, dict, None]]:
            # The number of retries left, starting at max_retries_for_selecting_speaker
            nonlocal attempts_left
            nonlocal attempt

            attempt = attempt + 1
            attempts_left = attempts_left - 1

            return self._validate_speaker_name(recipient, messages, sender, config, attempts_left, attempt, agents)

        # Two-agent chat for speaker selection

        # Agent for checking the response from the speaker_select_agent
        checking_agent, speaker_selection_agent = self._create_internal_agents(
            agents, max_attempts, messages, validate_speaker_name, selector
        )

        # Create the starting message
        if self.select_speaker_prompt_template is not None:
            start_message = {
                "content": self.select_speaker_prompt(agents),
                "override_role": self.role_for_select_speaker_messages,
            }
        else:
            start_message = messages[-1]

        # Add the message transforms, if any, to the speaker selection agent
        if self._speaker_selection_transforms is not None:
            self._speaker_selection_transforms.add_to_agent(speaker_selection_agent)

        # Run the speaker selection chat
        result = await checking_agent.a_initiate_chat(
            speaker_selection_agent,
            cache=None,  # don't use caching for the speaker selection chat
            message=start_message,
            max_turns=2
            * max(1, max_attempts),  # Limiting the chat to the number of attempts, including the initial one
            clear_history=False,
            silent=not self.select_speaker_auto_verbose,  # Base silence on the verbose attribute
        )

        return self._process_speaker_selection_result(result, last_speaker, agents)

    def _validate_speaker_name(
        self, recipient, messages, sender, config, attempts_left, attempt, agents
    ) -> tuple[bool, Union[str, dict, None]]:
        """Validates the speaker response for each round in the internal 2-agent
        chat within the  auto select speaker method.

        Used by auto_select_speaker and a_auto_select_speaker.
        """

        # Output the query and requery results
        if self.select_speaker_auto_verbose:
            iostream = IOStream.get_default()

        # Validate the speaker name selected
        select_name = messages[-1]["content"].strip()

        mentions = self._mentioned_agents(select_name, agents)

        if len(mentions) == 1:
            # Success on retry, we have just one name mentioned
            selected_agent_name = next(iter(mentions))

            # Add the selected agent to the response so we can return it
            messages.append({"role": "user", "content": f"[AGENT SELECTED]{selected_agent_name}"})

            if self.select_speaker_auto_verbose:
                iostream.print(
                    colored(
                        f">>>>>>>> Select speaker attempt {attempt} of {attempt + attempts_left} successfully selected: {selected_agent_name}",
                        "green",
                    ),
                    flush=True,
                )

        elif len(mentions) > 1:
            # More than one name on requery so add additional reminder prompt for next retry

            if self.select_speaker_auto_verbose:
                iostream.print(
                    colored(
                        f">>>>>>>> Select speaker attempt {attempt} of {attempt + attempts_left} failed as it included multiple agent names.",
                        "red",
                    ),
                    flush=True,
                )

            if attempts_left:
                # Message to return to the chat for the next attempt
                agentlist = f"{[agent.name for agent in agents]}"

                return True, {
                    "content": self.select_speaker_auto_multiple_template.format(agentlist=agentlist),
                    "name": "checking_agent",
                    "override_role": self.role_for_select_speaker_messages,
                }
            else:
                # Final failure, no attempts left
                messages.append(
                    {
                        "role": "user",
                        "content": f"[AGENT SELECTION FAILED]Select speaker attempt #{attempt} of {attempt + attempts_left} failed as it returned multiple names.",
                    }
                )

        else:
            # No names at all on requery so add additional reminder prompt for next retry

            if self.select_speaker_auto_verbose:
                iostream.print(
                    colored(
                        f">>>>>>>> Select speaker attempt #{attempt} failed as it did not include any agent names.",
                        "red",
                    ),
                    flush=True,
                )

            if attempts_left:
                # Message to return to the chat for the next attempt
                agentlist = f"{[agent.name for agent in agents]}"

                return True, {
                    "content": self.select_speaker_auto_none_template.format(agentlist=agentlist),
                    "name": "checking_agent",
                    "override_role": self.role_for_select_speaker_messages,
                }
            else:
                # Final failure, no attempts left
                messages.append(
                    {
                        "role": "user",
                        "content": f"[AGENT SELECTION FAILED]Select speaker attempt #{attempt} of {attempt + attempts_left} failed as it did not include any agent names.",
                    }
                )

        return True, None

    def _process_speaker_selection_result(self, result, last_speaker: ConversableAgent, agents: Optional[list[Agent]]):
        """Checks the result of the auto_select_speaker function, returning the
        agent to speak.

        Used by auto_select_speaker and a_auto_select_speaker."""
        if len(result.chat_history) > 0:
            # Use the final message, which will have the selected agent or reason for failure
            final_message = result.chat_history[-1]["content"]

            if "[AGENT SELECTED]" in final_message:
                # Have successfully selected an agent, return it
                return self.agent_by_name(final_message.replace("[AGENT SELECTED]", ""))

            else:  # "[AGENT SELECTION FAILED]"
                # Failed to select an agent, so we'll select the next agent in the list
                next_agent = self.next_agent(last_speaker, agents)

                # No agent, return the failed reason
                return next_agent

    def _participant_roles(self, agents: list[Agent] = None) -> str:
        # Default to all agents registered
        if agents is None:
            agents = self.agents

        roles = []
        for agent in agents:
            if agent.description.strip() == "":
                logger.warning(
                    f"The agent '{agent.name}' has an empty description, and may not work well with GroupChat."
                )
            roles.append(f"{agent.name}: {agent.description}".strip())
        return "\n".join(roles)

    def _mentioned_agents(self, message_content: Union[str, list], agents: Optional[list[Agent]]) -> dict:
        """Counts the number of times each agent is mentioned in the provided message content.
        Agent names will match under any of the following conditions (all case-sensitive):
        - Exact name match
        - If the agent name has underscores it will match with spaces instead (e.g. 'Story_writer' == 'Story writer')
        - If the agent name has underscores it will match with '\\_' instead of '_' (e.g. 'Story_writer' == 'Story\\_writer')

        Args:
            message_content (Union[str, List]): The content of the message, either as a single string or a list of strings.
            agents (List[Agent]): A list of Agent objects, each having a 'name' attribute to be searched in the message content.

        Returns:
            Dict: a counter for mentioned agents.
        """
        if agents is None:
            agents = self.agents

        # Cast message content to str
        if isinstance(message_content, dict):
            message_content = message_content["content"]
        message_content = content_str(message_content)

        mentions = dict()
        for agent in agents:
            # Finds agent mentions, taking word boundaries into account,
            # accommodates escaping underscores and underscores as spaces
            regex = (
                r"(?<=\W)("
                + re.escape(agent.name)
                + r"|"
                + re.escape(agent.name.replace("_", " "))
                + r"|"
                + re.escape(agent.name.replace("_", r"\_"))
                + r")(?=\W)"
            )
            count = len(re.findall(regex, f" {message_content} "))  # Pad the message to help with matching
            if count > 0:
                mentions[agent.name] = count
        return mentions