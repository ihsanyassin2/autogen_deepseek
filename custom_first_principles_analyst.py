import logging
import ollama
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from autogen.browser_utils import SimpleTextBrowser
from custom_conversable_agent import Agent, ConversableAgent

logger = logging.getLogger(__name__)

class FirstPrinciplesAnalyst(ConversableAgent):
    """Autonomous agent focused on complex problem solving and information gathering"""
    
    DEFAULT_PROMPT = """You are an autonomous problem-solving AI that can:
1. Break down complex problems into manageable subtasks
2. Think critically and analyze problems from multiple angles
3. Research and gather information independently
4. Generate comprehensive solutions and analysis
5. Adapt and revise approaches based on findings

Approach all tasks systematically and think step-by-step.
Use first principles thinking when breaking down problems."""

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
        
        # Initialize state
        self.current_task = None
        self.subtasks = []
        self.findings = {}
        
        # Initialize browser for research
        try:
            self.browser = SimpleTextBrowser()
        except Exception as e:
            logger.warning(f"Browser initialization failed: {e}")
            self.browser = None

    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """Main entry point for problem solving"""
        try:
            self.current_task = problem
            
            # 1. Break down the problem
            breakdown = self._break_down_problem(problem)
            if not breakdown["success"]:
                return breakdown
            
            # 2. Research and gather information
            research = self._gather_information(breakdown["data"])
            
            # 3. Analyze and synthesize solution
            solution = self._synthesize_solution(breakdown["data"], research["data"])
            
            return {
                "success": True,
                "data": {
                    "problem_breakdown": breakdown["data"],
                    "research_findings": research["data"],
                    "solution": solution["data"]
                }
            }
            
        except Exception as e:
            logger.error(f"Problem solving failed: {str(e)}")
            return {
                "success": False,
                "message": str(e)
            }

    

    def _query_llm(self, prompt: str) -> Dict[str, Any]:
        """Query Ollama/local LLM"""
        try:
            response = ollama.chat(
                model=self.llm_config.get('model', 'mistral'),
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            
            if not response or 'message' not in response:
                raise ValueError("Invalid LLM response format")
            
            return {
                "success": True,
                "data": response['message']['content']
            }
            
        except Exception as e:
            logger.error(f"LLM query failed: {str(e)}")
            return {
                "success": False,
                "message": f"LLM error: {str(e)}"
            }

    def _web_research(self, topic: str) -> Dict[str, Any]:
        """Perform web research on a topic"""
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
1. Key facts and information
2. Credible sources
3. Relevant insights
4. Latest developments
5. Important context

Focus on verified and relevant information."""

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

    def _extract_components(self, text: str) -> Dict[str, Any]:
        """Extract structured components from text"""
        components = {
            "core_components": [],
            "key_questions": [],
            "research_needed": [],
            "dependencies": [],
            "success_criteria": []
        }
        
        current_section = None
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.lower().startswith('core component'):
                current_section = "core_components"
            elif line.lower().startswith('question'):
                current_section = "key_questions"
            elif line.lower().startswith('research'):
                current_section = "research_needed"
            elif line.lower().startswith('depend'):
                current_section = "dependencies"
            elif line.lower().startswith('success'):
                current_section = "success_criteria"
            elif current_section and (line.startswith('-') or line.startswith('*')):
                components[current_section].append(line[1:].strip())
            
        return components
    
    def _break_down_problem(self, problem: str) -> Dict[str, Any]:
        """Break down problem to fundamental elements"""
        try:
            # First, identify the domain and core elements
            domain_prompt = f"""What is the fundamental domain/field this problem belongs to:
    Problem: {problem}

    Identify:
    1. Primary domain(s)
    2. Key subfields involved
    3. Basic principles that govern this domain
    4. Fundamental laws or rules that cannot be broken
    5. Absolute requirements that must be met

    Think from first principles. Strip away assumptions."""

            domain_analysis = self._query_llm(domain_prompt)

            # Then break down to core requirements
            requirements_prompt = f"""Based on this domain analysis:
    {domain_analysis["data"]}

    Break down the absolute core requirements for this problem:
    1. What MUST be true for any solution to work?
    2. What physical/real-world constraints exist?
    3. What resources are absolutely essential?
    4. What skills/capabilities are non-negotiable?
    5. What conditions must be met?

    Think at the most fundamental level possible."""

            requirements = self._query_llm(requirements_prompt)

            # Identify data needs
            data_prompt = f"""Given these core requirements:
    {requirements["data"]}

    What specific data do we need to validate each requirement:
    1. What metrics must we measure?
    2. What capabilities must we verify?
    3. What conditions must we confirm?
    4. What numbers do we need?
    5. What facts must we establish?

    Be specific about exact data needs."""

            data_needs = self._query_llm(data_prompt)

            # Structure the investigation
            investigation_prompt = f"""Plan the investigation:
    Domain Analysis: {domain_analysis["data"]}
    Core Requirements: {requirements["data"]}
    Data Needs: {data_needs["data"]}

    Create a structured plan to:
    1. Verify each requirement
    2. Gather each piece of needed data
    3. Validate each assumption
    4. Test each constraint
    5. Confirm each condition

    Organize from most fundamental to most dependent."""

            investigation_plan = self._query_llm(investigation_prompt)

            return {
                "success": True,
                "data": {
                    "domain_analysis": domain_analysis["data"],
                    "core_requirements": requirements["data"],
                    "data_needs": data_needs["data"],
                    "investigation_plan": investigation_plan["data"]
                }
            }

        except Exception as e:
            return {"success": False, "message": str(e)}

    def _gather_information(self, breakdown: Dict[str, Any]) -> Dict[str, Any]:
        """Gather information based on breakdown"""
        findings = {}
        
        try:
            # Extract data needs
            data_needs = breakdown["data_needs"]
            
            for need in data_needs:
                # First verify what's known for certain
                verification_prompt = f"""For this data need: {need}
                
    What can we verify with absolute certainty?
    What are the fundamental truths we know?
    What can be mathematically/logically proven?
    What physical laws apply?
    What is non-negotiable?"""

                verified = self._query_llm(verification_prompt)
                
                # Research factual data
                if self.browser:
                    factual = self._web_research(need)
                else:
                    factual = {"success": False, "data": None}
                    
                # Analyze gaps and uncertainties
                gaps_prompt = f"""Given:
    Verified facts: {verified["data"]}
    Research findings: {factual.get("data")}

    Identify:
    1. What remains uncertain?
    2. What assumptions are we making?
    3. What needs additional verification?
    4. What could invalidate our findings?
    5. What dependencies exist?"""

                gaps = self._query_llm(gaps_prompt)
                
                findings[need] = {
                    "verified_facts": verified["data"],
                    "research_findings": factual.get("data"),
                    "gaps_and_uncertainties": gaps["data"]
                }
                
        except Exception as e:
            return {"success": False, "message": str(e)}
            
        return {
            "success": True,
            "data": findings
        }

    def _synthesize_solution(self, breakdown: Dict[str, Any], findings: Dict[str, Any]) -> Dict[str, Any]:
        """Build solution from first principles"""
        try:
            # Validate core requirements are met
            validation_prompt = f"""Given our findings:
    {json.dumps(findings, indent=2)}

    Verify against core requirements:
    {json.dumps(breakdown["core_requirements"], indent=2)}

    For each requirement:
    1. Is it fully satisfied? Prove how.
    2. Is it partially met? Show evidence.
    3. Is it unmet? Explain why.
    4. What risks exist? Quantify them.
    5. What assumptions remain? List them."""

            validation = self._query_llm(validation_prompt)

            # Build solution from fundamentals up
            solution_prompt = f"""Based on validated requirements:
    {validation["data"]}

    Build a solution starting from absolute fundamentals:
    1. Start with verified facts only
    2. Add proven capabilities
    3. Layer in validated assumptions
    4. Address identified risks
    5. Fill gaps with research findings

    Show your work step by step."""

            solution = self._query_llm(solution_prompt)

            # Test solution against original requirements
            testing_prompt = f"""Test this solution:
    {solution["data"]}

    Against original requirements:
    {json.dumps(breakdown["core_requirements"], indent=2)}

    Verify:
    1. Does it meet all fundamental requirements?
    2. Does it violate any physical constraints?
    3. Does it depend on unproven assumptions?
    4. What could cause it to fail?
    5. How robust is it to changes?"""

            testing = self._query_llm(testing_prompt)

            return {
                "success": True,
                "data": {
                    "validation": validation["data"],
                    "solution": solution["data"],
                    "testing": testing["data"]
                }
            }

        except Exception as e:
            return {"success": False, "message": str(e)}
