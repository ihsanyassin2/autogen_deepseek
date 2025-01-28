import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
from autogen.browser_utils import SimpleTextBrowser
from custom_conversable_agent import ConversableAgent

logger = logging.getLogger(__name__)

class UnconventionalStrategist(ConversableAgent):
    """Autonomous agent focused on generating and evaluating unconventional strategies for problem-solving."""

    DEFAULT_PROMPT = """You are an autonomous strategist AI specializing in unconventional and innovative approaches to problem-solving.
1. Generate creative and unconventional strategies for the given problem.
2. Identify opportunities and leverage them for unique solutions.
3. Challenge traditional methods and propose alternative approaches.
4. Evaluate feasibility, scalability, and potential risks of innovative strategies.
5. Provide actionable steps to implement the most promising strategies.
Think disruptively and aim for innovative breakthroughs."""

    def __init__(
        self,
        name: str,
        system_message: str = DEFAULT_PROMPT,
        llm_config: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs
        )
        self.current_problem = None
        self.findings = {}

        # Initialize browser for research
        try:
            self.browser = SimpleTextBrowser()
        except Exception as e:
            logger.warning(f"Browser initialization failed: {e}")
            self.browser = None

    def strategize(self, problem: str) -> Dict[str, Any]:
        """Main method for generating and evaluating unconventional strategies."""
        try:
            self.current_problem = problem

            # Step 1: Generate unconventional strategies
            strategies = self._generate_strategies(problem)

            # Step 2: Evaluate opportunities
            opportunities = self._evaluate_opportunities(problem, strategies["data"])

            # Step 3: Assess risks and feasibility
            risk_assessment = self._assess_risks(strategies["data"])

            # Step 4: Provide actionable steps
            action_plan = self._provide_action_plan(strategies["data"], opportunities["data"], risk_assessment["data"])

            return {
                "success": True,
                "data": {
                    "strategies": strategies["data"],
                    "opportunities": opportunities["data"],
                    "risk_assessment": risk_assessment["data"],
                    "action_plan": action_plan["data"]
                }
            }

        except Exception as e:
            logger.error(f"Strategy generation failed: {str(e)}")
            return {
                "success": False,
                "message": str(e)
            }

    def _generate_strategies(self, problem: str) -> Dict[str, Any]:
        """Generate creative and unconventional strategies."""
        prompt = f"""Analyze this problem:
{problem}

Generate:
1. Unconventional and creative strategies.
2. Innovative approaches that deviate from traditional methods.
3. Unique angles or perspectives to solve the problem.
4. High-level concepts for breakthrough solutions.
"""
        return self._query_llm(prompt)

    def _evaluate_opportunities(self, problem: str, strategies: Any) -> Dict[str, Any]:
        """Identify opportunities and advantages for proposed strategies."""
        prompt = f"""Evaluate the following strategies for this problem:
{strategies}

Identify:
1. Opportunities that each strategy leverages.
2. Key advantages of these strategies compared to conventional approaches.
3. Potential for creating disruptive or breakthrough results.
"""
        return self._query_llm(prompt)

    def _assess_risks(self, strategies: Any) -> Dict[str, Any]:
        """Assess risks and feasibility of proposed strategies."""
        prompt = f"""Assess the following strategies:
{strategies}

Analyze:
1. Feasibility of implementation.
2. Potential risks or challenges for each strategy.
3. Limitations or dependencies that could hinder success.
4. Trade-offs between risks and potential rewards.
"""
        return self._query_llm(prompt)

    def _provide_action_plan(self, strategies: Any, opportunities: Any, risks: Any) -> Dict[str, Any]:
        """Provide actionable steps for implementing strategies."""
        prompt = f"""Based on the following analysis:
Strategies:
{strategies}

Opportunities:
{opportunities}

Risks:
{risks}

Provide:
1. A step-by-step action plan to implement the most promising strategies.
2. Recommendations to mitigate risks and maximize opportunities.
3. Strategies to ensure scalability and sustainability.
4. Practical steps to test and validate the effectiveness of the strategies.
"""
        return self._query_llm(prompt)

    def _query_llm(self, prompt: str) -> Dict[str, Any]:
        """Query the language model for analysis."""
        try:
            response = self.llm_config["model"].chat(
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            if not response or 'message' not in response:
                raise ValueError("Invalid LLM response format")

            return {
                "success": True,
                "data": response["message"]
            }

        except Exception as e:
            logger.error(f"LLM query failed: {str(e)}")
            return {
                "success": False,
                "message": str(e)
            }

    def _web_research(self, topic: str) -> Dict[str, Any]:
        """Perform web research on a topic."""
        try:
            if not self.browser:
                return {
                    "success": False,
                    "message": "Browser not available"
                }

            self.browser.visit_page(f"bing: {topic}")
            content = self.browser.viewport

            analysis_prompt = f"""Analyze this web content about {topic}:
{content}

Extract:
1. Key facts and information.
2. Credible sources.
3. Relevant insights.
4. Latest developments.
5. Important context.
"""
            analysis = self._query_llm(analysis_prompt)

            return {
                "success": True,
                "data": {
                    "raw_content": content,
                    "analysis": analysis["data"] if analysis["success"] else None
                }
            }

        except Exception as e:
            logger.error(f"Web research failed: {str(e)}")
            return {
                "success": False,
                "message": str(e)
            }
