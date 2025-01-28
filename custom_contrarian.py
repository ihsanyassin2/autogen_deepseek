import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
from autogen.browser_utils import SimpleTextBrowser
from custom_conversable_agent import ConversableAgent

logger = logging.getLogger(__name__)

class Contrarian(ConversableAgent):
    """Autonomous agent focused on challenging conventional ideas and exploring alternative viewpoints."""

    DEFAULT_PROMPT = """You are an autonomous contrarian AI that specializes in challenging conventional ideas and exploring alternative perspectives.
1. Question assumptions behind proposed ideas or solutions.
2. Identify overlooked or underexplored perspectives.
3. Generate alternative approaches that deviate from the norm.
4. Test the robustness of mainstream solutions by exposing weaknesses.
5. Provide insights that spark debate and deeper analysis.
Think independently and critically, aiming to uncover hidden flaws or opportunities."""

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

    def challenge_solution(self, solution: str) -> Dict[str, Any]:
        """Main method for challenging a proposed solution."""
        try:
            self.current_problem = solution

            # Step 1: Question assumptions
            assumptions = self._question_assumptions(solution)

            # Step 2: Explore alternative perspectives
            alternatives = self._explore_alternatives(solution)

            # Step 3: Highlight overlooked risks
            risks = self._highlight_risks(solution)

            # Step 4: Provide contrarian insights
            insights = self._provide_insights(solution, assumptions["data"], alternatives["data"], risks["data"])

            return {
                "success": True,
                "data": {
                    "assumptions": assumptions["data"],
                    "alternatives": alternatives["data"],
                    "risks": risks["data"],
                    "insights": insights["data"]
                }
            }

        except Exception as e:
            logger.error(f"Contrarian analysis failed: {str(e)}")
            return {
                "success": False,
                "message": str(e)
            }

    def _question_assumptions(self, solution: str) -> Dict[str, Any]:
        """Question the assumptions underlying the solution."""
        prompt = f"""Analyze this solution:
{solution}

Question:
1. What assumptions are being made, explicitly or implicitly?
2. Are these assumptions valid, or do they rely on weak foundations?
3. Could any assumptions fail under different conditions?
4. How would changing these assumptions alter the solution?
"""
        return self._query_llm(prompt)

    def _explore_alternatives(self, solution: str) -> Dict[str, Any]:
        """Explore alternative approaches to the solution."""
        prompt = f"""Critique this solution:
{solution}

Explore:
1. Alternative approaches that challenge the conventional path.
2. Uncommon methods or ideas that could address the same problem.
3. Perspectives that deviate from the mainstream thinking.
4. Innovative concepts that might outperform the proposed solution.
"""
        return self._query_llm(prompt)

    def _highlight_risks(self, solution: str) -> Dict[str, Any]:
        """Highlight overlooked risks or vulnerabilities in the solution."""
        prompt = f"""Evaluate this solution:
{solution}

Identify:
1. Hidden risks or vulnerabilities that may have been overlooked.
2. Scenarios where the solution could fail.
3. Dependencies that could introduce uncertainty.
4. Trade-offs that may compromise the solution’s effectiveness.
"""
        return self._query_llm(prompt)

    def _provide_insights(self, solution: str, assumptions: Any, alternatives: Any, risks: Any) -> Dict[str, Any]:
        """Provide contrarian insights to spark debate and analysis."""
        prompt = f"""Based on the following analysis:
Solution:
{solution}

Assumptions:
{assumptions}

Alternatives:
{alternatives}

Risks:
{risks}

Provide:
1. Contrarian insights that challenge the conventional narrative.
2. Unique perspectives that provoke deeper analysis.
3. Suggestions for refining or rethinking the solution.
4. Questions that need to be answered to strengthen the proposal.
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
