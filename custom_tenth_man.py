import logging
import ollama
import json
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from autogen.browser_utils import SimpleTextBrowser
from custom_conversable_agent import Agent, ConversableAgent
from custom_ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class TenthManAgent(ConversableAgent):
    """Autonomous contrarian analyst agent with enhanced research capabilities"""
    
    DEFAULT_PROMPT = """You are The Tenth Man - an autonomous contrarian analyst. Your role:
1. Question assumptions and consensus
2. Research contrarian viewpoints independently  
3. Surface overlooked risks and challenges
4. Challenge conventional thinking
5. Provide evidence-based counterarguments

Today's date is """ + datetime.now().date().isoformat()

    def __init__(
        self,
        name: str,
        system_message: str = DEFAULT_PROMPT,
        llm_config: Optional[dict] = None,
        browser_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=system_message, 
            llm_config=llm_config,
            **kwargs,
        )
        
        # Initialize components with enhanced validation
        if not browser_config:
            browser_config = {}  # Empty dict for default settings
        self.browser = SimpleTextBrowser(**browser_config)
        self.client = OllamaClient(config=llm_config) if llm_config else None
        
        # State tracking
        self.current_topic = None
        self.research_history = []
        
        # Register core capabilities
        self._register_functions()

    def _register_functions(self):
        """Register autonomous research functions"""
        functions = {
            "research_topic": self._research_topic,
            "analyze_evidence": self._analyze_evidence,
            "extract_insights": self._extract_insights,
            "identify_risks": self._identify_risks
        }
        self.register_function(functions)

    def generate_reply(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        sender: Optional[Agent] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any], None]:
        """Autonomous research and analysis workflow that builds on previous agent insights."""
        try:
            # If messages is not provided, use agent's chat history
            if messages is None:
                messages = self.chat_messages.get(sender, [])
            
            # Step 1: Validate 'messages'
            if not messages:
                logger.error("No messages provided to generate_reply.")
                return self._error_response("No messages provided. A non-empty list of messages is required.")

            if not isinstance(messages, list):
                logger.error(f"Messages is not a list: {type(messages)}")
                return self._error_response("Messages must be a list of dictionaries.")

            if not all(isinstance(m, dict) for m in messages):
                logger.error("One or more items in messages is not a dictionary.")
                return self._error_response("Each message must be a dictionary.")

            logger.debug(f"Validated input messages: {json.dumps(messages, indent=2)}")

            # Step 2: Validate each message
            for idx, message in enumerate(messages):
                if "content" not in message:
                    logger.error(f"Message at index {idx} is missing 'content': {message}")
                    return self._error_response(f"Message at index {idx} is missing 'content'.")

                if not isinstance(message["content"], str):
                    logger.error(f"Message at index {idx} has non-string 'content': {message}")
                    return self._error_response(f"Message at index {idx} must have 'content' as a string.")

                if not message["content"].strip():
                    logger.error(f"Message at index {idx} has empty 'content': {message}")
                    return self._error_response(f"Message at index {idx} has empty 'content'.")

                if "role" not in message:
                    logger.warning(f"Message at index {idx} is missing 'role'. Assigning default role 'user'.")
                    message["role"] = "user"

            # Step 3: Extract the latest topic/message
            topic = messages[-1].get("content", "").strip()
            if not topic:
                logger.error("The last message does not contain valid content.")
                return self._error_response("The last message must contain valid content.")

            self.current_topic = topic
            logger.debug(f"Extracted topic: {topic}")

            # Step 4: Analyze previous messages
            try:
                previous_analyses = self._analyze_previous_messages(messages[:-1])
                if not previous_analyses:
                    logger.warning("No previous analyses found or analyses are improperly formatted.")
                    previous_analyses = {}
                logger.debug(f"Previous analyses: {json.dumps(previous_analyses, indent=2)}")
            except Exception as e:
                logger.error(f"Error analyzing previous messages: {str(e)}")
                return self._error_response("Failed to analyze previous messages.")

            # Step 5: Generate research questions
            try:
                questions_prompt = self._analyze_topic(topic, previous_analyses)
                research_questions = self._query_llm(questions_prompt)

                if not research_questions.get("success"):
                    logger.error("Failed to generate research questions.")
                    return self._error_response("Failed to generate research questions.")

                logger.debug(f"Generated research questions: {json.dumps(research_questions, indent=2)}")
            except Exception as e:
                logger.error(f"Error generating research questions: {str(e)}")
                return self._error_response("Failed to generate research questions.")

            # Step 6: Conduct research on questions
            evidence = []
            try:
                for question in self._parse_questions(research_questions["data"].get("response", "")):
                    research = self._research_topic(question)
                    if research.get("success"):
                        evidence.append({
                            "question": question,
                            "findings": research["data"],
                            "related_analyses": self._find_related_analyses(question, previous_analyses)
                        })
                        logger.info(f"Research completed for question: {question}")

                if not evidence:
                    logger.warning("No evidence was collected during research phase.")
                logger.debug(f"Collected evidence: {json.dumps(evidence, indent=2)}")
            except Exception as e:
                logger.error(f"Error during research phase: {str(e)}")
                return self._error_response("An error occurred while conducting research.")

            # Step 7: Analyze findings
            try:
                analysis = self._analyze_evidence(evidence, previous_analyses)
                if not analysis.get("success"):
                    logger.error("Failed to analyze evidence.")
                    return self._error_response("Failed to analyze evidence.")
                logger.debug(f"Analysis results: {json.dumps(analysis, indent=2)}")
            except Exception as e:
                logger.error(f"Error analyzing evidence: {str(e)}")
                return self._error_response("An error occurred while analyzing evidence.")

            # Step 8: Extract insights
            try:
                insights = self._extract_insights(analysis.get("data", ""), previous_analyses)
                if not insights.get("success"):
                    logger.warning("Failed to extract insights.")
                    insights = {"data": "No insights extracted."}
                logger.debug(f"Extracted insights: {json.dumps(insights, indent=2)}")
            except Exception as e:
                logger.error(f"Error extracting insights: {str(e)}")
                insights = {"data": "Error extracting insights."}

            # Step 9: Identify risks
            try:
                risks = self._identify_risks(evidence, previous_analyses)
                if not risks.get("success"):
                    logger.warning("Failed to identify risks.")
                    risks = {"data": "No risks identified."}
                logger.debug(f"Identified risks: {json.dumps(risks, indent=2)}")
            except Exception as e:
                logger.error(f"Error identifying risks: {str(e)}")
                risks = {"data": "Error identifying risks."}

            # Step 10: Format and return response
            try:
                response = self._format_response(
                    evidence=evidence,
                    analysis=analysis, 
                    insights=insights,
                    risks=risks,
                    previous_analyses=previous_analyses
                )
                logger.debug(f"Formatted response: {json.dumps(response, indent=2)}")
                return response
            except Exception as e:
                logger.error(f"Error formatting response: {str(e)}")
                return self._error_response("An error occurred while formatting the response.")

        except Exception as e:
            logger.error(f"Unexpected error in generate_reply: {str(e)}", exc_info=True)
            return self._error_response(f"An unexpected error occurred: {str(e)}")





    def _analyze_previous_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze previous messages from other agents with enhanced validation"""
        agent_messages = [
            m for m in messages 
            if m.get("role") == "assistant" 
            and m.get("name") != self.name
            and isinstance(m.get("content"), str)
        ]

        insights = {}
        for msg in agent_messages:
            try:
                content = msg["content"].strip()
                if not content:
                    continue
                    
                insights[msg["name"]] = {
                    "content": content,
                    "key_points": self._extract_key_points(content)
                }
            except Exception as e:
                logger.debug(f"Error processing message from {msg.get('name')}: {str(e)}")
                continue

        return insights or {"default": {"content": "No valid previous analyses", "key_points": []}}

    def _analyze_topic(self, topic: str, previous_analyses: Dict[str, Any]) -> str:
        """Generate analysis prompt with conditional previous context"""
        base_prompt = """Analyze this topic:
        
    Develop new questions that:
    1. Challenge unconsidered assumptions
    2. Explore gaps in current analysis
    3. Test team's consensus views
    4. Identify overlooked risks
    5. Question proposed solutions

    Format questions on new lines with numbers."""
        
        if previous_analyses:
            return f"""Analyze this topic building on previous team insights:

    Previous analyses:
    {json.dumps(previous_analyses, indent=2)}

    {base_prompt}"""
        
        return base_prompt


    def _find_related_analyses(self, question: str, previous_analyses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find relevant previous analyses with fallback handling"""
        if not isinstance(previous_analyses, dict):
            return []

        related = []
        for agent, analysis in previous_analyses.items():
            content = analysis.get("content", "")
            if not content:
                continue
                
            prompt = f"""Compare relevance between:
    Question: {question}
    Analysis: {content}
    Respond ONLY with 'related' or 'unrelated'"""
            
            response = self._query_llm(prompt)
            if response.get("success") and "related" in response["data"]["response"].lower():
                related.append({
                    "agent": agent,
                    "points": analysis.get("key_points", [])
                })

        return related or [{"agent": "System", "points": ["No related analyses found"]}]



    def _extract_key_points(self, content: str) -> List[str]:
        prompt = f"Extract key analytical points from: {content}"
        response = self._query_llm(prompt)
        return response.get("data", {}).get("response", "").split("\n")


    
    
    def _research_topic(self, question: str) -> Dict[str, Any]:
        """Execute full research cycle for a question"""
        try:
            logger.info(f"🔍 Researching: {question}")
            
            # Web research phase
            web_results = self._web_search(question)
            if not web_results.get("success"):
                return {"success": False, "message": "Web research failed"}

            # Analysis phase
            analysis_prompt = f"""Critically evaluate this evidence:
Question: {question}
Evidence: {web_results['data']['raw_results']}

Consider:
1. Source credibility
2. Evidence quality
3. Missing perspectives
4. Alternative explanations
5. Potential biases"""

            analysis = self._query_llm(analysis_prompt)
            
            return {
                "success": True,
                "data": {
                    "question": question,
                    "web_results": web_results["data"],
                    "analysis": analysis.get("data", {}).get("response", "")
                }
            }

        except Exception as e:
            logger.error(f"Research failure: {str(e)}")
            return {"success": False, "message": str(e)}

    def _web_search(self, query: str) -> Dict[str, Any]:
        """Execute web search with error handling"""
        try:
            if not hasattr(self, 'browser') or not self.browser:
                logger.error("Browser is not initialized.")
                return {"success": False, "message": "Research browser unavailable.", "data": {"query": query}}

            logger.info(f"🌐 Searching: {query}")
            self.browser.visit_page(f"duckduckgo: {query}")
            content = self.browser.viewport

            if not content:
                logger.warning("No content retrieved from web search.")
                return {"success": False, "message": "No results from web search.", "data": {"query": query}}

            return {"success": True, "data": {"query": query, "raw_results": content}}

        except Exception as e:
            logger.error(f"Web search failed: {str(e)}", exc_info=True)
            return {"success": False, "message": f"Search failed: {str(e)}"}


    def _analyze_evidence(self, evidence: List[Dict[str, Any]], previous_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize findings with conditional previous context"""
        analysis_prompt = """Synthesize findings:
    Evidence:
    {evidence}
    Identify:
    1. New insights
    2. Contradictions
    3. Emerging patterns
    4. Unaddressed areas
    5. Critical implications"""
        
        if previous_analyses:
            analysis_prompt = """Synthesize findings and previous analyses:
    Evidence:
    {evidence}
    Previous Team Analyses:
    {previous_analyses}
    Identify:
    1. New insights vs previous findings
    2. Contradictions between analyses
    3. Emerging patterns
    4. Unaddressed areas
    5. Critical implications"""
        
        return self._query_llm(analysis_prompt.format(
            evidence=json.dumps(evidence, indent=2),
            previous_analyses=json.dumps(previous_analyses, indent=2)
        ))

    def _extract_insights(self, analysis: str, previous_analyses: Dict[str, Any]) -> Dict[str, Any]:
        try:
            insight_prompt = f"""Extract insights integrating current and previous analyses:
    {analysis}
    Previous Analyses: {json.dumps(previous_analyses, indent=2)}
    Focus on:
    1. Hidden connections
    2. Systemic impacts
    3. Future scenarios
    4. Strategic implications
    5. Innovation opportunities"""
            
            insights = self._query_llm(insight_prompt)
            return {
                "success": True,
                "data": insights.get("data", {}).get("response", "")
            }
        except Exception as e:
            logger.error(f"Insight extraction failed: {str(e)}")
            return {"success": False, "message": str(e)}

    def _identify_risks(self, evidence: List[Dict[str, Any]], previous_analyses: Dict[str, Any]) -> Dict[str, Any]:
        try:
            risk_prompt = f"""Assess risks considering all evidence:
    Evidence: {json.dumps(evidence, indent=2)}
    Previous Analyses: {json.dumps(previous_analyses, indent=2)}
    Consider:
    1. Direct operational risks
    2. Systemic vulnerabilities 
    3. Long-term consequences
    4. Hidden assumptions
    5. Black swan potentials"""

            risks = self._query_llm(risk_prompt)
            return {
                "success": True,
                "data": risks.get("data", {}).get("response", "")
            }
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            return {"success": False, "message": str(e)}

    def _query_llm(self, prompt: str) -> Dict[str, Any]:
        """Execute LLM query with logging"""
        try:
            if not prompt:  # Add null check
                logger.error("Empty prompt provided to LLM")
                return {"success": False, "message": "Empty prompt"}

            model = self.llm_config.get("config_list", [{}])[0].get("model", "mistral")
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]


            logger.debug(f"🤖 LLM Query: {prompt[:200]}...")
            response = ollama.chat(model=model, messages=messages)

            if not response or "message" not in response:
                logger.error("LLM response is empty or malformed.")
                return {"success": False, "message": "Invalid LLM response structure."}

            content = self._extract_message_content(response)

            if not content:
                logger.warning("LLM response contained no content.")
                return {"success": False, "message": "No content in LLM response."}

            return {"success": True, "data": {"response": content}}

        except Exception as e:
            logger.error(f"LLM query failed: {str(e)}", exc_info=True)
            return {"success": False, "message": f"Query failed: {str(e)}"}


    def _extract_message_content(self, response: Any) -> str:
        """Robust content extraction"""
        try:
            if isinstance(response, dict):
                message = response.get('message')
                if isinstance(message, dict):
                    return message.get('content', '')
                elif isinstance(message, str):
                    return message
            elif hasattr(response, 'message'):
                return getattr(response.message, 'content', str(response.message))

            logger.warning("Response content could not be extracted.")
            return ''

        except Exception as e:
            logger.error(f"Content extraction error: {str(e)}", exc_info=True)
            return ''


    def _parse_questions(self, raw_text: str) -> List[str]:
        """Parse research questions from text"""
        return [q.split('. ', 1)[-1] for q in raw_text.split("\n") if q.strip()]

    def _format_response(
        self,
        evidence: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        insights: Dict[str, Any],
        risks: Dict[str, Any],
        previous_analyses: Dict[str, Any]
    ) -> Dict[str, str]:
        """Structure final response with robust type checking and error handling"""
        response = ["**Tenth Man Analysis Report**\n"]
        
        # Section 1: Research Findings
        try:
            if evidence:
                response.append("\n**🔍 Key Research Findings:**")
                for idx, e in enumerate(evidence):
                    # Safe extraction with type checking
                    findings = e.get('findings') or {}
                    if not isinstance(findings, dict):
                        findings = {}
                    
                    analysis_text = str(findings.get('analysis', '⚠️ Preliminary analysis pending'))
                    analysis_preview = analysis_text[:300] + ("..." if len(analysis_text) > 300 else "")
                    
                    response.append(f"\n• **Question:** {e.get('question', f'Research Question #{idx+1}')}")
                    response.append(f"  **Analysis:** {analysis_preview}")
                    
                    # Handle related analyses with list safety
                    related = e.get('related_analyses', [])
                    if isinstance(related, list) and related:
                        response.append("  **Related Insights:**")
                        for rel in related:
                            agent_name = rel.get('agent', 'Unknown Source')
                            points = rel.get('points', [])
                            point_text = str(points[0])[:200] + "..." if (isinstance(points, list) and points) else "No key points"
                            response.append(f"    - {agent_name}: {point_text}")
                    else:
                        response.append("  **Related Insights:** No relevant connections found")
        except Exception as e:
            logger.error(f"Research findings error: {str(e)}")
            response.append("\n⚠️ Partial research findings unavailable due to processing errors")

        # Section 2: Synthesis
        synthesis_content = "🔍 Synthesis pending - initial analysis required"
        try:
            if isinstance(analysis, dict) and analysis.get("success"):
                analysis_data = analysis.get("data", {})
                if isinstance(analysis_data, dict):
                    synthesis_text = str(analysis_data.get("response", "⚠️ Incomplete synthesis"))
                    synthesis_content = synthesis_text[:500] + ("..." if len(synthesis_text) > 500 else "")
        except Exception as e:
            logger.error(f"Synthesis error: {str(e)}")
            synthesis_content = "⚠️ Synthesis data corrupted"
        response.append("\n**🧩 Synthesis:**\n" + synthesis_content)

        # Section 3: Strategic Insights
        insights_content = "🔭 Insights generation pending further analysis"
        try:
            if isinstance(insights, dict) and insights.get("success"):
                insights_data = insights.get("data", {})
                if isinstance(insights_data, dict):
                    insights_text = str(insights_data.get("response", "⚠️ Partial insights"))
                    insights_content = insights_text[:500] + ("..." if len(insights_text) > 500 else "")
        except Exception as e:
            logger.error(f"Insights error: {str(e)}")
            insights_content = "⚠️ Insights data corrupted"
        response.append("\n**💡 Strategic Insights:**\n" + insights_content)

        # Section 4: Risk Assessment
        risks_content = "⚠️ Risk assessment incomplete - additional research required"
        try:
            if isinstance(risks, dict) and risks.get("success"):
                risks_data = risks.get("data", {})
                if isinstance(risks_data, dict):
                    risks_text = str(risks_data.get("response", "⚠️ Partial risk assessment"))
                    risks_content = risks_text[:500] + ("..." if len(risks_text) > 500 else "")
        except Exception as e:
            logger.error(f"Risks error: {str(e)}")
            risks_content = "⚠️ Risk data corrupted"
        response.append("\n**⚠️ Risk Assessment:**\n" + risks_content)

        # Dynamic recommendation
        try:
            has_content = any([
                bool(evidence),
                (isinstance(analysis, dict) and analysis.get("success")),
                (isinstance(insights, dict) and insights.get("success"))
            ])
            recommendation = ("\n**✅ Recommendation:** Validate through independent verification" 
                            if has_content else
                            "\n**⚠️ Recommendation:** Initial analysis insufficient - expand research scope")
            response.append(recommendation)
        except Exception as e:
            logger.error(f"Recommendation error: {str(e)}")
            response.append("\n**⚠️ Recommendation:** Analysis incomplete - verify input data")

        return {
            "role": "assistant",
            "content": "\n".join(response).strip()
        }

    def _error_response(self, error: str) -> Dict[str, str]:
        """Standard error formatting"""
        return {
            "role": "assistant",
            "content": f"🔴 Analysis Error: {error}\nPlease verify inputs and try again."
        }