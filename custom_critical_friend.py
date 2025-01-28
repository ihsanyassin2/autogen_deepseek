import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
from autogen.browser_utils import SimpleTextBrowser
from custom_conversable_agent import ConversableAgent

logger = logging.getLogger(__name__)

class CriticalFriend(ConversableAgent):
    """Autonomous agent focused on constructive feedback and validation of proposed solutions."""

    DEFAULT_PROMPT = """You are an autonomous constructive feedback AI that specializes in validating ideas, plans, or solutions by offering supportive critique.
1. Identify strengths and positive aspects of the solution.
2. Highlight potential areas for improvement constructively.
3. Verify assumptions and dependencies.
4. Ensure alignment with objectives and goals.
5. Provide actionable recommendations to refine and enhance the solution.
Think collaboratively and focus on constructive development."""

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

    def validate_solution(self, solution: str) -> Dict[str, Any]:
        """Main method for validating a solution."""
        try:
            self.current_solution = solution

            # Step 1: Identify strengths
            strengths = self._identify_strengths(solution)

            # Step 2: Highlight improvement areas
            improvement_areas = self._highlight_improvement_areas(solution)

            # Step 3: Verify assumptions and dependencies
            assumptions = self._verify_assumptions(solution)

            # Step 4: Provide actionable recommendations
            recommendations = self._provide_recommendations(solution)

            return {
                "success": True,
                "data": {
                    "strengths": strengths["data"],
                    "improvement_areas": improvement_areas["data"],
                    "assumptions": assumptions["data"],
                    "recommendations": recommendations["data"]
                }
            }

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                "success": False,
                "message": str(e)
            }

    def _identify_strengths(self, solution: str) -> Dict[str, Any]:
        """Identify strengths and positive aspects of the solution."""
        prompt = f"""Analyze this solution:
{solution}

Identify:
1. Key strengths and positive attributes.
2. Elements that align well with objectives.
3. Aspects that add significant value.
4. Factors that contribute to feasibility and success.
"""
        return self._query_llm(prompt)

    def _highlight_improvement_areas(self, solution: str) -> Dict[str, Any]:
        """Highlight areas for improvement constructively."""
        prompt = f"""Evaluate this solution:
{solution}

Highlight:
1. Potential areas for improvement.
2. Elements that could be refined or optimized.
3. Dependencies or assumptions that need further validation.
4. Risks or challenges that should be addressed.
"""
        return self._query_llm(prompt)

    def _verify_assumptions(self, solution: str) -> Dict[str, Any]:
        """Verify assumptions and dependencies in the solution."""
        prompt = f"""Analyze this solution:
{solution}

Verify:
1. Key assumptions and whether they are valid.
2. Dependencies and their reliability.
3. Consistency with known data and facts.
4. Alignment with the stated goals and objectives.
"""
        return self._query_llm(prompt)

    def _provide_recommendations(self, solution: str) -> Dict[str, Any]:
        """Provide actionable recommendations to refine the solution."""
        prompt = f"""Based on the analysis of this solution:
{solution}

Provide:
1. Specific recommendations to address improvement areas.
2. Strategies to strengthen strengths and minimize risks.
3. Suggestions for validating assumptions and dependencies.
4. Steps to enhance overall feasibility and success.
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
