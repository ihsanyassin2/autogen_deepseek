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
from custom_groupchat import GroupChat

from custom_conversable_agent import ConversableAgent

logger = logging.getLogger(__name__)

class GroupChatManager(ConversableAgent):
    """(In preview) A chat manager agent that can manage a group chat of multiple agents."""

    def __init__(
        self,
        groupchat: GroupChat,
        name: Optional[str] = "chat_manager",
        # unlimited consecutive auto reply by default
        max_consecutive_auto_reply: Optional[int] = sys.maxsize,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "NEVER",
        system_message: Optional[Union[str, list]] = "Group chat manager.",
        silent: bool = False,
        **kwargs,
    ):
        if (
            kwargs.get("llm_config")
            and isinstance(kwargs["llm_config"], dict)
            and (kwargs["llm_config"].get("functions") or kwargs["llm_config"].get("tools"))
        ):
            raise ValueError(
                "GroupChatManager is not allowed to make function/tool calls. Please remove the 'functions' or 'tools' config in 'llm_config' you passed in."
            )

        super().__init__(
            name=name,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            system_message=system_message,
            **kwargs,
        )
        if logging_enabled():
            log_new_agent(self, locals())
        # Store groupchat
        self._groupchat = groupchat

        self._last_speaker = None
        self._silent = silent

        # Order of register_reply is important.
        # Allow sync chat if initiated using initiate_chat
        self.register_reply(Agent, GroupChatManager.run_chat, config=groupchat, reset_config=GroupChat.reset)
        # Allow async chat if initiated using a_initiate_chat
        self.register_reply(
            Agent,
            GroupChatManager.a_run_chat,
            config=groupchat,
            reset_config=GroupChat.reset,
            ignore_async_in_sync_chat=True,
        )

    @property
    def groupchat(self) -> GroupChat:
        """Returns the group chat managed by the group chat manager."""
        return self._groupchat

    def chat_messages_for_summary(self, agent: Agent) -> list[dict]:
        """The list of messages in the group chat as a conversation to summarize.
        The agent is ignored.
        """
        return self._groupchat.messages

    def _prepare_chat(
        self,
        recipient: ConversableAgent,
        clear_history: bool,
        prepare_recipient: bool = True,
        reply_at_receive: bool = True,
    ) -> None:
        super()._prepare_chat(recipient, clear_history, prepare_recipient, reply_at_receive)

        if clear_history:
            self._groupchat.reset()

        for agent in self._groupchat.agents:
            if (recipient != agent or prepare_recipient) and isinstance(agent, ConversableAgent):
                agent._prepare_chat(self, clear_history, False, reply_at_receive)

    @property
    def last_speaker(self) -> Agent:
        """Return the agent who sent the last message to group chat manager.

        In a group chat, an agent will always send a message to the group chat manager, and the group chat manager will
        send the message to all other agents in the group chat. So, when an agent receives a message, it will always be
        from the group chat manager. With this property, the agent receiving the message can know who actually sent the
        message.

        Example:
        ```python
        from autogen import ConversableAgent
        from autogen import GroupChat, GroupChatManager


        def print_messages(recipient, messages, sender, config):
            # Print the message immediately
            print(
                f"Sender: {sender.name} | Recipient: {recipient.name} | Message: {messages[-1].get('content')}"
            )
            print(f"Real Sender: {sender.last_speaker.name}")
            assert sender.last_speaker.name in messages[-1].get("content")
            return False, None  # Required to ensure the agent communication flow continues


        agent_a = ConversableAgent("agent A", default_auto_reply="I'm agent A.")
        agent_b = ConversableAgent("agent B", default_auto_reply="I'm agent B.")
        agent_c = ConversableAgent("agent C", default_auto_reply="I'm agent C.")
        for agent in [agent_a, agent_b, agent_c]:
            agent.register_reply(
                [ConversableAgent, None], reply_func=print_messages, config=None
            )
        group_chat = GroupChat(
            [agent_a, agent_b, agent_c],
            messages=[],
            max_round=6,
            speaker_selection_method="random",
            allow_repeat_speaker=True,
        )
        chat_manager = GroupChatManager(group_chat)
        groupchat_result = agent_a.initiate_chat(
            chat_manager, message="Hi, there, I'm agent A."
        )
        ```
        """
        return self._last_speaker

    def run_chat(
        self,
        messages: Optional[list[dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[GroupChat] = None,
    ) -> tuple[bool, Optional[str]]:
        """Run a group chat."""
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        speaker = sender
        groupchat = config
        send_introductions = getattr(groupchat, "send_introductions", False)
        silent = getattr(self, "_silent", False)

        if send_introductions:
            # Broadcast the intro
            intro = groupchat.introductions_msg()
            for agent in groupchat.agents:
                self.send(intro, agent, request_reply=False, silent=True)
            # NOTE: We do not also append to groupchat.messages,
            # since groupchat handles its own introductions

        if self.client_cache is not None:
            for a in groupchat.agents:
                a.previous_cache = a.client_cache
                a.client_cache = self.client_cache
        for i in range(groupchat.max_round):
            self._last_speaker = speaker
            groupchat.append(message, speaker)
            # broadcast the message to all agents except the speaker
            for agent in groupchat.agents:
                if agent != speaker:
                    self.send(message, agent, request_reply=False, silent=True)
            if self._is_termination_msg(message) or i == groupchat.max_round - 1:
                # The conversation is over or it's the last round
                break
            try:
                # select the next speaker
                speaker = groupchat.select_speaker(speaker, self)
                if not silent:
                    iostream = IOStream.get_default()
                    iostream.print(colored(f"\nNext speaker: {speaker.name}\n", "green"), flush=True)
                # let the speaker speak
                reply = speaker.generate_reply(sender=self)
            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = speaker.generate_reply(sender=self)
                else:
                    # admin agent is not found in the participants
                    raise
            except NoEligibleSpeaker:
                # No eligible speaker, terminate the conversation
                break

            if reply is None:
                # no reply is generated, exit the chat
                break

            # check for "clear history" phrase in reply and activate clear history function if found
            if (
                groupchat.enable_clear_history
                and isinstance(reply, dict)
                and reply["content"]
                and "CLEAR HISTORY" in reply["content"].upper()
            ):
                reply["content"] = self.clear_agents_history(reply, groupchat)

            # The speaker sends the message without requesting a reply
            speaker.send(reply, self, request_reply=False, silent=silent)
            message = self.last_message(speaker)
        if self.client_cache is not None:
            for a in groupchat.agents:
                a.client_cache = a.previous_cache
                a.previous_cache = None
        return True, None

    async def a_run_chat(
        self,
        messages: Optional[list[dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[GroupChat] = None,
    ):
        """Run a group chat asynchronously."""
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        speaker = sender
        groupchat = config
        send_introductions = getattr(groupchat, "send_introductions", False)
        silent = getattr(self, "_silent", False)

        if send_introductions:
            # Broadcast the intro
            intro = groupchat.introductions_msg()
            for agent in groupchat.agents:
                await self.a_send(intro, agent, request_reply=False, silent=True)
            # NOTE: We do not also append to groupchat.messages,
            # since groupchat handles its own introductions

        if self.client_cache is not None:
            for a in groupchat.agents:
                a.previous_cache = a.client_cache
                a.client_cache = self.client_cache
        for i in range(groupchat.max_round):
            groupchat.append(message, speaker)

            if self._is_termination_msg(message):
                # The conversation is over
                break

            # broadcast the message to all agents except the speaker
            for agent in groupchat.agents:
                if agent != speaker:
                    await self.a_send(message, agent, request_reply=False, silent=True)
            if i == groupchat.max_round - 1:
                # the last round
                break
            try:
                # select the next speaker
                speaker = await groupchat.a_select_speaker(speaker, self)
                # let the speaker speak
                reply = await speaker.a_generate_reply(sender=self)
            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = await speaker.a_generate_reply(sender=self)
                else:
                    # admin agent is not found in the participants
                    raise
            if reply is None:
                break
            # The speaker sends the message without requesting a reply
            await speaker.a_send(reply, self, request_reply=False, silent=silent)
            message = self.last_message(speaker)
        if self.client_cache is not None:
            for a in groupchat.agents:
                a.client_cache = a.previous_cache
                a.previous_cache = None
        return True, None

    def resume(
        self,
        messages: Union[list[dict], str],
        remove_termination_string: Optional[Union[str, Callable[[str], str]]] = None,
        silent: Optional[bool] = False,
    ) -> tuple[ConversableAgent, dict]:
        """Resumes a group chat using the previous messages as a starting point. Requires the agents, group chat, and group chat manager to be established
        as per the original group chat.

        Args:
            - messages Union[List[Dict], str]: The content of the previous chat's messages, either as a Json string or a list of message dictionaries.
            - remove_termination_string (str or function): Remove the termination string from the last message to prevent immediate termination
                If a string is provided, this string will be removed from last message.
                If a function is provided, the last message will be passed to this function.
            - silent (bool or None): (Experimental) whether to print the messages for this conversation. Default is False.

        Returns:
            - Tuple[ConversableAgent, Dict]: A tuple containing the last agent who spoke and their message
        """

        # Convert messages from string to messages list, if needed
        if isinstance(messages, str):
            messages = self.messages_from_string(messages)
        elif isinstance(messages, list) and all(isinstance(item, dict) for item in messages):
            messages = copy.deepcopy(messages)
        else:
            raise Exception("Messages is not of type str or List[Dict]")

        # Clean up the objects, ensuring there are no messages in the agents and group chat

        # Clear agent message history
        for agent in self._groupchat.agents:
            if isinstance(agent, ConversableAgent):
                agent.clear_history()

        # Clear Manager message history
        self.clear_history()

        # Clear GroupChat messages
        self._groupchat.reset()

        # Validation of message and agents

        try:
            self._valid_resume_messages(messages)
        except:
            raise

        # Load the messages into the group chat
        for i, message in enumerate(messages):
            if "name" in message:
                message_speaker_agent = self._groupchat.agent_by_name(message["name"])
            else:
                # If there's no name, assign the group chat manager (this is an indication the ChatResult messages was used instead of groupchat.messages as state)
                message_speaker_agent = self
                message["name"] = self.name

            # If it wasn't an agent speaking, it may be the manager
            if not message_speaker_agent and message["name"] == self.name:
                message_speaker_agent = self

            # Add previous messages to each agent (except the last message, as we'll kick off the conversation with it)
            if i != len(messages) - 1:
                for agent in self._groupchat.agents:
                    if agent.name == message["name"]:
                        # An agent`s message is sent to the Group Chat Manager
                        agent.send(message, self, request_reply=False, silent=True)
                    else:
                        # Otherwise, messages are sent from the Group Chat Manager to the agent
                        self.send(message, agent, request_reply=False, silent=True)

                # Add previous message to the new groupchat, if it's an admin message the name may not match so add the message directly
                if message_speaker_agent:
                    self._groupchat.append(message, message_speaker_agent)
                else:
                    self._groupchat.messages.append(message)

            # Last speaker agent
            last_speaker_name = message["name"]

            # Last message to check for termination (we could avoid this by ignoring termination check for resume in the future)
            last_message = message

        # Get last speaker as an agent
        previous_last_agent = self._groupchat.agent_by_name(name=last_speaker_name)

        # If we didn't match a last speaker agent, we check that it's the group chat's admin name and assign the manager, if so
        if not previous_last_agent and (
            last_speaker_name == self._groupchat.admin_name or last_speaker_name == self.name
        ):
            previous_last_agent = self

        # Termination removal and check
        self._process_resume_termination(remove_termination_string, messages)

        if not silent:
            iostream = IOStream.get_default()
            iostream.print(
                f"Prepared group chat with {len(messages)} messages, the last speaker is",
                colored(last_speaker_name, "yellow"),
                flush=True,
            )

        # Update group chat settings for resuming
        self._groupchat.send_introductions = False

        return previous_last_agent, last_message

    async def a_resume(
        self,
        messages: Union[list[dict], str],
        remove_termination_string: Optional[Union[str, Callable[[str], str]]] = None,
        silent: Optional[bool] = False,
    ) -> tuple[ConversableAgent, dict]:
        """Resumes a group chat using the previous messages as a starting point, asynchronously. Requires the agents, group chat, and group chat manager to be established
        as per the original group chat.

        Args:
            - messages Union[List[Dict], str]: The content of the previous chat's messages, either as a Json string or a list of message dictionaries.
            - remove_termination_string (str or function): Remove the termination string from the last message to prevent immediate termination
                If a string is provided, this string will be removed from last message.
                If a function is provided, the last message will be passed to this function, and the function returns the string after processing.
            - silent (bool or None): (Experimental) whether to print the messages for this conversation. Default is False.

        Returns:
            - Tuple[ConversableAgent, Dict]: A tuple containing the last agent who spoke and their message
        """

        # Convert messages from string to messages list, if needed
        if isinstance(messages, str):
            messages = self.messages_from_string(messages)
        elif isinstance(messages, list) and all(isinstance(item, dict) for item in messages):
            messages = copy.deepcopy(messages)
        else:
            raise Exception("Messages is not of type str or List[Dict]")

        # Clean up the objects, ensuring there are no messages in the agents and group chat

        # Clear agent message history
        for agent in self._groupchat.agents:
            if isinstance(agent, ConversableAgent):
                agent.clear_history()

        # Clear Manager message history
        self.clear_history()

        # Clear GroupChat messages
        self._groupchat.reset()

        # Validation of message and agents

        try:
            self._valid_resume_messages(messages)
        except:
            raise

        # Load the messages into the group chat
        for i, message in enumerate(messages):
            if "name" in message:
                message_speaker_agent = self._groupchat.agent_by_name(message["name"])
            else:
                # If there's no name, assign the group chat manager (this is an indication the ChatResult messages was used instead of groupchat.messages as state)
                message_speaker_agent = self
                message["name"] = self.name

            # If it wasn't an agent speaking, it may be the manager
            if not message_speaker_agent and message["name"] == self.name:
                message_speaker_agent = self

            # Add previous messages to each agent (except the last message, as we'll kick off the conversation with it)
            if i != len(messages) - 1:
                for agent in self._groupchat.agents:
                    if agent.name == message["name"]:
                        # An agent`s message is sent to the Group Chat Manager
                        agent.a_send(message, self, request_reply=False, silent=True)
                    else:
                        # Otherwise, messages are sent from the Group Chat Manager to the agent
                        self.a_send(message, agent, request_reply=False, silent=True)

                # Add previous message to the new groupchat, if it's an admin message the name may not match so add the message directly
                if message_speaker_agent:
                    self._groupchat.append(message, message_speaker_agent)
                else:
                    self._groupchat.messages.append(message)

            # Last speaker agent
            last_speaker_name = message["name"]

            # Last message to check for termination (we could avoid this by ignoring termination check for resume in the future)
            last_message = message

        # Get last speaker as an agent
        previous_last_agent = self._groupchat.agent_by_name(name=last_speaker_name)

        # If we didn't match a last speaker agent, we check that it's the group chat's admin name and assign the manager, if so
        if not previous_last_agent and (
            last_speaker_name == self._groupchat.admin_name or last_speaker_name == self.name
        ):
            previous_last_agent = self

        # Termination removal and check
        self._process_resume_termination(remove_termination_string, messages)

        if not silent:
            iostream = IOStream.get_default()
            iostream.print(
                f"Prepared group chat with {len(messages)} messages, the last speaker is",
                colored(last_speaker_name, "yellow"),
                flush=True,
            )

        # Update group chat settings for resuming
        self._groupchat.send_introductions = False

        return previous_last_agent, last_message

    def _valid_resume_messages(self, messages: list[dict]):
        """Validates the messages used for resuming

        args:
            messages (List[Dict]): list of messages to resume with

        returns:
            - bool: Whether they are valid for resuming
        """
        # Must have messages to start with, otherwise they should run run_chat
        if not messages:
            raise Exception(
                "Cannot resume group chat as no messages were provided. Use GroupChatManager.run_chat or ConversableAgent.initiate_chat to start a new chat."
            )

        # Check that all agents in the chat messages exist in the group chat
        for message in messages:
            if message.get("name"):
                if (
                    not self._groupchat.agent_by_name(message["name"])
                    and not message["name"] == self._groupchat.admin_name  # ignore group chat's name
                    and not message["name"] == self.name  # ignore group chat manager's name
                ):
                    raise Exception(f"Agent name in message doesn't exist as agent in group chat: {message['name']}")

    def _process_resume_termination(
        self, remove_termination_string: Union[str, Callable[[str], str]], messages: list[dict]
    ):
        """Removes termination string, if required, and checks if termination may occur.

        args:
            remove_termination_string (str or function): Remove the termination string from the last message to prevent immediate termination
                If a string is provided, this string will be removed from last message.
                If a function is provided, the last message will be passed to this function, and the function returns the string after processing.

        returns:
            None
        """

        last_message = messages[-1]

        # Replace any given termination string in the last message
        if isinstance(remove_termination_string, str):

            def _remove_termination_string(content: str) -> str:
                return content.replace(remove_termination_string, "")

        else:
            _remove_termination_string = remove_termination_string

        if _remove_termination_string:
            if messages[-1].get("content"):
                messages[-1]["content"] = _remove_termination_string(messages[-1]["content"])

        # Check if the last message meets termination (if it has one)
        if self._is_termination_msg:
            if self._is_termination_msg(last_message):
                logger.warning("WARNING: Last message meets termination criteria and this may terminate the chat.")

    def messages_from_string(self, message_string: str) -> list[dict]:
        """Reads the saved state of messages in Json format for resume and returns as a messages list

        args:
            - message_string: Json string, the saved state

        returns:
            - List[Dict]: List of messages
        """
        try:
            state = json.loads(message_string)
        except json.JSONDecodeError:
            raise Exception("Messages string is not a valid JSON string")

        return state

    def messages_to_string(self, messages: list[dict]) -> str:
        """Converts the provided messages into a Json string that can be used for resuming the chat.
        The state is made up of a list of messages

        args:
            - messages (List[Dict]): set of messages to convert to a string

        returns:
            - str: Json representation of the messages which can be persisted for resuming later
        """

        return json.dumps(messages)

    def _raise_exception_on_async_reply_functions(self) -> None:
        """Raise an exception if any async reply functions are registered.

        Raises:
            RuntimeError: if any async reply functions are registered.
        """
        super()._raise_exception_on_async_reply_functions()

        for agent in self._groupchat.agents:
            agent._raise_exception_on_async_reply_functions()

    def clear_agents_history(self, reply: dict, groupchat: GroupChat) -> str:
        """Clears history of messages for all agents or selected one. Can preserve selected number of last messages.
        That function is called when user manually provide "clear history" phrase in his reply.
        When "clear history" is provided, the history of messages for all agents is cleared.
        When "clear history <agent_name>" is provided, the history of messages for selected agent is cleared.
        When "clear history <nr_of_messages_to_preserve>" is provided, the history of messages for all agents is cleared
        except last <nr_of_messages_to_preserve> messages.
        When "clear history <agent_name> <nr_of_messages_to_preserve>" is provided, the history of messages for selected
        agent is cleared except last <nr_of_messages_to_preserve> messages.
        Phrase "clear history" and optional arguments are cut out from the reply before it passed to the chat.

        Args:
            reply (dict): reply message dict to analyze.
            groupchat (GroupChat): GroupChat object.
        """
        iostream = IOStream.get_default()

        reply_content = reply["content"]
        # Split the reply into words
        words = reply_content.split()
        # Find the position of "clear" to determine where to start processing
        clear_word_index = next(i for i in reversed(range(len(words))) if words[i].upper() == "CLEAR")
        # Extract potential agent name and steps
        words_to_check = words[clear_word_index + 2 : clear_word_index + 4]
        nr_messages_to_preserve = None
        nr_messages_to_preserve_provided = False
        agent_to_memory_clear = None

        for word in words_to_check:
            if word.isdigit():
                nr_messages_to_preserve = int(word)
                nr_messages_to_preserve_provided = True
            elif word[:-1].isdigit():  # for the case when number of messages is followed by dot or other sign
                nr_messages_to_preserve = int(word[:-1])
                nr_messages_to_preserve_provided = True
            else:
                for agent in groupchat.agents:
                    if agent.name == word:
                        agent_to_memory_clear = agent
                        break
                    elif agent.name == word[:-1]:  # for the case when agent name is followed by dot or other sign
                        agent_to_memory_clear = agent
                        break
        # preserve last tool call message if clear history called inside of tool response
        if "tool_responses" in reply and not nr_messages_to_preserve:
            nr_messages_to_preserve = 1
            logger.warning(
                "The last tool call message will be saved to prevent errors caused by tool response without tool call."
            )
        # clear history
        if agent_to_memory_clear:
            if nr_messages_to_preserve:
                iostream.print(
                    f"Clearing history for {agent_to_memory_clear.name} except last {nr_messages_to_preserve} messages."
                )
            else:
                iostream.print(f"Clearing history for {agent_to_memory_clear.name}.")
            agent_to_memory_clear.clear_history(nr_messages_to_preserve=nr_messages_to_preserve)
        else:
            if nr_messages_to_preserve:
                iostream.print(f"Clearing history for all agents except last {nr_messages_to_preserve} messages.")
                # clearing history for groupchat here
                temp = groupchat.messages[-nr_messages_to_preserve:]
                groupchat.messages.clear()
                groupchat.messages.extend(temp)
            else:
                iostream.print("Clearing history for all agents.")
                # clearing history for groupchat here
                groupchat.messages.clear()
            # clearing history for agents
            for agent in groupchat.agents:
                agent.clear_history(nr_messages_to_preserve=nr_messages_to_preserve)

        # Reconstruct the reply without the "clear history" command and parameters
        skip_words_number = 2 + int(bool(agent_to_memory_clear)) + int(nr_messages_to_preserve_provided)
        reply_content = " ".join(words[:clear_word_index] + words[clear_word_index + skip_words_number :])

        return reply_content
