import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
from autogen.browser_utils import SimpleTextBrowser
from custom_conversable_agent import ConversableAgent

logger = logging.getLogger(__name__)

class DevilsAdvocate(ConversableAgent):
    """Autonomous agent focused on critical analysis and identifying flaws in proposed solutions."""

    DEFAULT_PROMPT = """You are an autonomous critical analysis AI that specializes in finding flaws and weaknesses in ideas, plans, or solutions.
1. Critically evaluate every aspect of a proposed solution.
2. Identify assumptions, dependencies, and risks.
3. Challenge ideas with logical reasoning and counterexamples.
4. Provide alternative perspectives and approaches.
5. Ensure robustness and resilience by addressing vulnerabilities.
Think systematically and ensure no stone is left unturned."""

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
        self.current_solution = None
        self.findings = {}

        # Initialize browser for research
        try:
            self.browser = SimpleTextBrowser()
        except Exception as e:
            logger.warning(f"Browser initialization failed: {e}")
            self.browser = None

    def critique_solution(self, solution: str) -> Dict[str, Any]:
        """Main method for critiquing a solution."""
        try:
            self.current_solution = solution

            # Step 1: Identify assumptions and dependencies
            assumptions = self._identify_assumptions(solution)

            # Step 2: Analyze risks and vulnerabilities
            risks = self._analyze_risks(solution)

            # Step 3: Challenge with counterexamples
            challenges = self._generate_counterexamples(solution)

            # Step 4: Suggest improvements and alternatives
            suggestions = self._suggest_improvements(solution)

            return {
                "success": True,
                "data": {
                    "assumptions": assumptions["data"],
                    "risks": risks["data"],
                    "challenges": challenges["data"],
                    "suggestions": suggestions["data"]
                }
            }

        except Exception as e:
            logger.error(f"Critique failed: {str(e)}")
            return {
                "success": False,
                "message": str(e)
            }

    def _identify_assumptions(self, solution: str) -> Dict[str, Any]:
        """Identify assumptions and dependencies in the solution."""
        prompt = f"""Analyze this solution:
{solution}

Identify:
1. Implicit and explicit assumptions.
2. Dependencies and interdependencies.
3. Unverified or questionable premises.
4. Potential blind spots or oversights.
"""
        return self._query_llm(prompt)

    def _analyze_risks(self, solution: str) -> Dict[str, Any]:
        """Analyze risks and vulnerabilities in the solution."""
        prompt = f"""Evaluate this solution:
{solution}

Identify:
1. Potential risks and vulnerabilities.
2. Scenarios where the solution could fail.
3. Unintended consequences or side effects.
4. Long-term sustainability and scalability issues.
"""
        return self._query_llm(prompt)

    def _generate_counterexamples(self, solution: str) -> Dict[str, Any]:
        """Generate counterexamples to challenge the solution."""
        prompt = f"""Critically challenge this solution:
{solution}

Generate:
1. Logical counterexamples that expose weaknesses.
2. Contradictory scenarios that invalidate assumptions.
3. Cases where the solution does not apply or breaks down.
4. Examples of similar failed approaches in the past.
"""
        return self._query_llm(prompt)

    def _suggest_improvements(self, solution: str) -> Dict[str, Any]:
        """Suggest improvements or alternative approaches."""
        prompt = f"""Based on the identified flaws in this solution:
{solution}

Provide:
1. Improvements to address vulnerabilities and risks.
2. Alternative approaches to achieve the same goal.
3. Ways to validate assumptions and strengthen the solution.
4. Strategies to make the solution more robust and resilient.
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
