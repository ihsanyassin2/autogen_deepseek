import logging
import ollama
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from custom_conversable_agent import Agent, ConversableAgent

logger = logging.getLogger(__name__)

class CriticAgent(ConversableAgent):
    """An agent that provides critical analysis and feedback using Ollama.
    
    This agent can:
    1. Analyze proposals and strategies for potential issues and risks
    2. Evaluate feasibility and effectiveness of plans
    3. Provide constructive feedback and suggestions for improvement
    4. Identify gaps and inconsistencies in arguments
    5. Assess resource allocation and timeline realism
    """

    DEFAULT_PROMPT = """You are a thoughtful and constructive Critic Agent with expertise in strategic analysis.
Your role is to (whenever applicable):
1. Carefully analyze proposals, strategies and plans
2. Identify potential issues, risks and implementation challenges
3. Evaluate feasibility and effectiveness
4. Provide specific, actionable feedback for improvement
5. Highlight gaps in logic or missing considerations
6. Assess resource allocation and timeline realism

When analyzing content:
- Take a balanced approach, acknowledging both strengths and weaknesses
- Support critiques with clear reasoning and evidence
- Provide constructive suggestions, not just criticism
- Consider practical implementation challenges
- Evaluate alignment with stated goals and objectives
- Assess resource requirements and constraints

Your feedback should be:
- Specific and actionable
- Well-reasoned and evidence-based 
- Balanced and constructive
- Forward-looking and solution-oriented

Today's date is """ + datetime.now().date().isoformat()

    def __init__(
        self,
        name: str,
        system_message: str = DEFAULT_PROMPT,
        llm_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs,
        )
        
        # Initialize critic state
        self.current_analysis = None
        self.analysis_history = []
        self.feedback_status = {}
        
        # Register critic functions
        self._register_critic_functions()

    def _register_critic_functions(self):
        """Register core critic functions"""
        function_map = {
            "analyze_proposal": self._analyze_proposal,
            "evaluate_feasibility": self._evaluate_feasibility,
            "provide_feedback": self._provide_feedback,
            "assess_risks": self._assess_risks,
            "first_principles_analysis": self._first_principles_analysis,
            "critical_friend_review": self._critical_friend_review,
            "devils_advocate": self._devils_advocate,
            "tenth_man_analysis": self._tenth_man_analysis,
            "objective_adherence_check": self._objective_adherence_check,
            "contrarian_analysis": self._contrarian_analysis
        }
        
        self.register_function(function_map)
        
        # Register tools for LLM if config exists
        if self.llm_config:
            tools = [
                {
                    "function": {
                        "name": "analyze_proposal",
                        "description": "Analyze a proposal and identify potential issues",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "proposal": {
                                    "type": "string",
                                    "description": "The proposal to analyze"
                                }
                            },
                            "required": ["proposal"]
                        }
                    }
                },
                {
                    "function": {
                        "name": "evaluate_feasibility",
                        "description": "Evaluate the feasibility of a proposed plan",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "plan": {
                                    "type": "string",
                                    "description": "The plan to evaluate"
                                }
                            },
                            "required": ["plan"]
                        }
                    }
                }
            ]
            
            for tool in tools:
                self.update_tool_signature(tool, is_remove=False)

    def _analyze_proposal(self, proposal: str) -> Dict[str, Any]:
        """Analyze a proposal and identify potential issues and areas for improvement."""
        try:
            messages = [
                {"role": "system", "content": "Analyze this proposal and identify key issues, risks, and areas for improvement."},
                {"role": "user", "content": proposal}
            ]
            
            model = self.llm_config.get("config_list", [{}])[0].get("model", "mistral")
            response = ollama.chat(model=model, messages=messages)
            
            analysis = self._extract_message_content(response)
            
            return {
                "success": True,
                "message": "Proposal analysis completed",
                "data": {
                    "analysis": analysis,
                    "issues": self._extract_issues(analysis),
                    "suggestions": self._extract_suggestions(analysis)
                }
            }
        except Exception as e:
            logger.error(f"Proposal analysis error: {str(e)}")
            return {
                "success": False,
                "message": f"Analysis failed: {str(e)}",
                "data": None
            }

    def _evaluate_feasibility(self, plan: str) -> Dict[str, Any]:
        """Evaluate the feasibility of a proposed plan."""
        try:
            messages = [
                {"role": "system", "content": "Evaluate the feasibility of this plan, considering resources, timeline, and implementation challenges."},
                {"role": "user", "content": plan}
            ]
            
            model = self.llm_config.get("config_list", [{}])[0].get("model", "mistral")
            response = ollama.chat(model=model, messages=messages)
            
            evaluation = self._extract_message_content(response)
            
            return {
                "success": True,
                "message": "Feasibility evaluation completed",
                "data": {
                    "evaluation": evaluation,
                    "challenges": self._extract_challenges(evaluation),
                    "recommendations": self._extract_recommendations(evaluation)
                }
            }
        except Exception as e:
            logger.error(f"Feasibility evaluation error: {str(e)}")
            return {
                "success": False,
                "message": f"Evaluation failed: {str(e)}",
                "data": None
            }

    def _provide_feedback(self, content: str) -> Dict[str, Any]:
        """Provide constructive feedback on content."""
        try:
            messages = [
                {"role": "system", "content": "Provide specific, constructive feedback on this content, highlighting both strengths and areas for improvement."},
                {"role": "user", "content": content}
            ]
            
            model = self.llm_config.get("config_list", [{}])[0].get("model", "mistral")
            response = ollama.chat(model=model, messages=messages)
            
            feedback = self._extract_message_content(response)
            
            # Store feedback
            feedback_entry = {
                "feedback": feedback,
                "timestamp": datetime.now().isoformat(),
                "content": content
            }
            self.analysis_history.append(feedback_entry)
            
            return {
                "success": True,
                "message": "Feedback provided",
                "data": feedback_entry
            }
        except Exception as e:
            logger.error(f"Feedback generation error: {str(e)}")
            return {
                "success": False,
                "message": f"Feedback failed: {str(e)}",
                "data": None
            }

    def _assess_risks(self, proposal: str) -> Dict[str, Any]:
        """Assess potential risks and challenges in a proposal."""
        try:
            messages = [
                {"role": "system", "content": "Identify and assess potential risks, challenges, and mitigation strategies for this proposal."},
                {"role": "user", "content": proposal}
            ]
            
            model = self.llm_config.get("config_list", [{}])[0].get("model", "mistral")
            response = ollama.chat(model=model, messages=messages)
            
            assessment = self._extract_message_content(response)
            
            return {
                "success": True,
                "message": "Risk assessment completed",
                "data": {
                    "assessment": assessment,
                    "risks": self._extract_risks(assessment),
                    "mitigations": self._extract_mitigations(assessment)
                }
            }
        except Exception as e:
            logger.error(f"Risk assessment error: {str(e)}")
            return {
                "success": False,
                "message": f"Assessment failed: {str(e)}",
                "data": None
            }

    def _extract_message_content(self, response: Any) -> str:
        """Safely extract message content from Ollama response."""
        try:
            if isinstance(response, dict):
                message = response.get('message')
                if hasattr(message, 'content'):
                    return message.content
                elif isinstance(message, dict):
                    return message.get('content', '')
                elif isinstance(message, str):
                    return message
            elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                return response.message.content
            elif hasattr(response, 'message') and isinstance(response.message, dict):
                return response.message.get('content', '')
            return ''
        except Exception as e:
            logger.error(f"Error extracting message content: {str(e)}")
            return ''

    def _extract_issues(self, analysis: str) -> List[str]:
        """Extract identified issues from analysis text."""
        issues = []
        lines = analysis.split('\n')
        
        for line in lines:
            if any(marker in line.lower() for marker in ['issue:', 'problem:', 'concern:', 'challenge:']):
                issues.append(line.strip())
                
        return issues

    def _extract_suggestions(self, analysis: str) -> List[str]:
        """Extract suggestions from analysis text."""
        suggestions = []
        lines = analysis.split('\n')
        
        for line in lines:
            if any(marker in line.lower() for marker in ['suggest:', 'recommend:', 'could:', 'should:']):
                suggestions.append(line.strip())
                
        return suggestions

    def _extract_challenges(self, evaluation: str) -> List[str]:
        """Extract implementation challenges from evaluation text."""
        challenges = []
        lines = evaluation.split('\n')
        
        for line in lines:
            if any(marker in line.lower() for marker in ['challenge:', 'difficulty:', 'obstacle:', 'barrier:']):
                challenges.append(line.strip())
                
        return challenges

    def _extract_recommendations(self, evaluation: str) -> List[str]:
        """Extract recommendations from evaluation text."""
        recommendations = []
        lines = evaluation.split('\n')
        
        for line in lines:
            if any(marker in line.lower() for marker in ['recommend:', 'suggest:', 'advise:', 'propose:']):
                recommendations.append(line.strip())
                
        return recommendations

    def _extract_risks(self, assessment: str) -> List[str]:
        """Extract identified risks from assessment text."""
        risks = []
        lines = assessment.split('\n')
        
        for line in lines:
            if any(marker in line.lower() for marker in ['risk:', 'threat:', 'vulnerability:', 'exposure:']):
                risks.append(line.strip())
                
        return risks

    def _extract_mitigations(self, assessment: str) -> List[str]:
        """Extract mitigation strategies from assessment text."""
        mitigations = []
        lines = assessment.split('\n')
        
        for line in lines:
            if any(marker in line.lower() for marker in ['mitigation:', 'solution:', 'strategy:', 'control:']):
                mitigations.append(line.strip())
                
        return mitigations

    def _first_principles_analysis(self, content: str) -> Dict[str, Any]:
        """Analyze content using first principles thinking.
        
        Breaks down complex problems into fundamental elements and rebuilds from there."""
        try:
            messages = [
                {"role": "system", "content": """Using first principles thinking:
                1. Break down the complex problem into fundamental elements
                2. Eliminate assumptions
                3. Identify basic truths and essential components
                4. Rebuild the solution from foundational elements"""},
                {"role": "user", "content": content}
            ]
            
            model = self.llm_config.get("config_list", [{}])[0].get("model", "mistral")
            response = ollama.chat(model=model, messages=messages)
            
            analysis = self._extract_message_content(response)
            
            return {
                "success": True,
                "message": "First principles analysis completed",
                "data": {
                    "analysis": analysis,
                    "fundamental_elements": self._extract_fundamentals(analysis),
                    "assumptions_challenged": self._extract_assumptions(analysis),
                    "core_truths": self._extract_core_truths(analysis)
                }
            }
        except Exception as e:
            logger.error(f"First principles analysis error: {str(e)}")
            return {
                "success": False,
                "message": f"Analysis failed: {str(e)}",
                "data": None
            }

    def _critical_friend_review(self, content: str) -> Dict[str, Any]:
        """Review content as a critical friend - supportive yet challenging."""
        try:
            messages = [
                {"role": "system", "content": """As a critical friend:
                1. Provide supportive but honest feedback
                2. Challenge assumptions constructively
                3. Ask probing questions
                4. Offer alternative perspectives
                5. Maintain a positive, growth-oriented mindset"""},
                {"role": "user", "content": content}
            ]
            
            model = self.llm_config.get("config_list", [{}])[0].get("model", "mistral")
            response = ollama.chat(model=model, messages=messages)
            
            review = self._extract_message_content(response)
            
            return {
                "success": True,
                "message": "Critical friend review completed",
                "data": {
                    "review": review,
                    "probing_questions": self._extract_questions(review),
                    "alternative_perspectives": self._extract_perspectives(review),
                    "constructive_suggestions": self._extract_suggestions(review)
                }
            }
        except Exception as e:
            logger.error(f"Critical friend review error: {str(e)}")
            return {
                "success": False,
                "message": f"Review failed: {str(e)}",
                "data": None
            }

    def _devils_advocate(self, content: str) -> Dict[str, Any]:
        """Take a devil's advocate position to identify potential flaws and challenges."""
        try:
            messages = [
                {"role": "system", "content": """As devil's advocate:
                1. Actively challenge assumptions and proposals
                2. Identify potential failure points
                3. Question underlying logic
                4. Expose hidden weaknesses
                5. Propose counter-arguments"""},
                {"role": "user", "content": content}
            ]
            
            model = self.llm_config.get("config_list", [{}])[0].get("model", "mistral")
            response = ollama.chat(model=model, messages=messages)
            
            analysis = self._extract_message_content(response)
            
            return {
                "success": True,
                "message": "Devil's advocate analysis completed",
                "data": {
                    "analysis": analysis,
                    "challenges": self._extract_challenges(analysis),
                    "counter_arguments": self._extract_counter_arguments(analysis),
                    "failure_points": self._extract_failure_points(analysis)
                }
            }
        except Exception as e:
            logger.error(f"Devil's advocate analysis error: {str(e)}")
            return {
                "success": False,
                "message": f"Analysis failed: {str(e)}",
                "data": None
            }

    def _tenth_man_analysis(self, content: str) -> Dict[str, Any]:
        """Implement the tenth man rule - deliberately disagree with consensus to prevent groupthink."""
        try:
            messages = [
                {"role": "system", "content": """As the tenth man:
                1. Deliberately challenge group consensus
                2. Explore unconsidered alternatives
                3. Question popular assumptions
                4. Identify overlooked risks
                5. Propose unconventional perspectives"""},
                {"role": "user", "content": content}
            ]
            
            model = self.llm_config.get("config_list", [{}])[0].get("model", "mistral")
            response = ollama.chat(model=model, messages=messages)
            
            analysis = self._extract_message_content(response)
            
            return {
                "success": True,
                "message": "Tenth man analysis completed",
                "data": {
                    "analysis": analysis,
                    "alternative_views": self._extract_alternatives(analysis),
                    "overlooked_risks": self._extract_risks(analysis),
                    "unconventional_approaches": self._extract_unconventional(analysis)
                }
            }
        except Exception as e:
            logger.error(f"Tenth man analysis error: {str(e)}")
            return {
                "success": False,
                "message": f"Analysis failed: {str(e)}",
                "data": None
            }

    def _objective_adherence_check(self, content: str) -> Dict[str, Any]:
        """Check how well proposals adhere to stated objectives and requirements."""
        try:
            messages = [
                {"role": "system", "content": """Evaluate objective adherence:
                1. Extract stated objectives and requirements
                2. Assess alignment of proposals with objectives
                3. Identify gaps in meeting requirements
                4. Measure potential effectiveness
                5. Suggest alignment improvements"""},
                {"role": "user", "content": content}
            ]
            
            model = self.llm_config.get("config_list", [{}])[0].get("model", "mistral")
            response = ollama.chat(model=model, messages=messages)
            
            analysis = self._extract_message_content(response)
            
            return {
                "success": True,
                "message": "Objective adherence check completed",
                "data": {
                    "analysis": analysis,
                    "alignment_assessment": self._extract_alignment(analysis),
                    "gaps": self._extract_gaps(analysis),
                    "effectiveness_measures": self._extract_effectiveness(analysis)
                }
            }
        except Exception as e:
            logger.error(f"Objective adherence check error: {str(e)}")
            return {
                "success": False,
                "message": f"Check failed: {str(e)}",
                "data": None
            }

    def _contrarian_analysis(self, content: str) -> Dict[str, Any]:
        """Take a contrarian view to challenge conventional thinking."""
        try:
            messages = [
                {"role": "system", "content": """As a contrarian:
                1. Challenge conventional wisdom
                2. Propose opposite approaches
                3. Question underlying assumptions
                4. Identify alternative paradigms
                5. Suggest unconventional solutions"""},
                {"role": "user", "content": content}
            ]
            
            model = self.llm_config.get("config_list", [{}])[0].get("model", "mistral")
            response = ollama.chat(model=model, messages=messages)
            
            analysis = self._extract_message_content(response)
            
            return {
                "success": True,
                "message": "Contrarian analysis completed",
                "data": {
                    "analysis": analysis,
                    "opposite_approaches": self._extract_opposite_approaches(analysis),
                    "challenged_assumptions": self._extract_assumptions(analysis),
                    "unconventional_solutions": self._extract_unconventional(analysis)
                }
            }
        except Exception as e:
            logger.error(f"Contrarian analysis error: {str(e)}")
            return {
                "success": False,
                "message": f"Analysis failed: {str(e)}",
                "data": None
            }

    # Helper methods for extracting specific types of content
    def _extract_fundamentals(self, text: str) -> List[str]:
        """Extract fundamental elements from analysis."""
        fundamentals = []
        lines = text.split('\n')
        for line in lines:
            if any(marker in line.lower() for marker in ['fundamental:', 'basic:', 'essential:', 'core:']):
                fundamentals.append(line.strip())
        return fundamentals

    def _extract_assumptions(self, text: str) -> List[str]:
        """Extract challenged assumptions from analysis."""
        assumptions = []
        lines = text.split('\n')
        for line in lines:
            if any(marker in line.lower() for marker in ['assumption:', 'presume:', 'assume:', 'given:']):
                assumptions.append(line.strip())
        return assumptions

    def _extract_core_truths(self, text: str) -> List[str]:
        """Extract core truths from analysis."""
        truths = []
        lines = text.split('\n')
        for line in lines:
            if any(marker in line.lower() for marker in ['truth:', 'fact:', 'principle:', 'axiom:']):
                truths.append(line.strip())
        return truths

    def _extract_questions(self, text: str) -> List[str]:
        """Extract probing questions from review."""
        questions = []
        lines = text.split('\n')
        for line in lines:
            if '?' in line:
                questions.append(line.strip())
        return questions

    def _extract_perspectives(self, text: str) -> List[str]:
        """Extract alternative perspectives from review."""
        perspectives = []
        lines = text.split('\n')
        for line in lines:
            if any(marker in line.lower() for marker in ['alternatively:', 'perspective:', 'view:', 'consider:']):
                perspectives.append(line.strip())
        return perspectives

    def _extract_counter_arguments(self, text: str) -> List[str]:
        """Extract counter-arguments from analysis."""
        counter_args = []
        lines = text.split('\n')
        for line in lines:
            if any(marker in line.lower() for marker in ['however:', 'contrary:', 'opposite:', 'instead:']):
                counter_args.append(line.strip())
        return counter_args

    def _extract_failure_points(self, text: str) -> List[str]:
        """Extract potential failure points from analysis."""
        failures = []
        lines = text.split('\n')
        for line in lines:
            if any(marker in line.lower() for marker in ['fail:', 'breakdown:', 'weakness:', 'vulnerable:']):
                failures.append(line.strip())
        return failures

    def _extract_alternatives(self, text: str) -> List[str]:
        """Extract alternative approaches from analysis."""
        alternatives = []
        lines = text.split('\n')
        for line in lines:
            if any(marker in line.lower() for marker in ['alternative:', 'another:', 'different:', 'other:']):
                alternatives.append(line.strip())
        return alternatives

    def _extract_unconventional(self, text: str) -> List[str]:
        """Extract unconventional approaches from analysis."""
        unconventional = []
        lines = text.split('\n')
        for line in lines:
            if any(marker in line.lower() for marker in ['unconventional:', 'novel:', 'unique:', 'innovative:']):
                unconventional.append(line.strip())
        return unconventional

    def _extract_alignment(self, text: str) -> List[str]:
        """Extract objective alignment assessments."""
        alignments = []
        lines = text.split('\n')
        for line in lines:
            if any(marker in line.lower() for marker in ['align:', 'match:', 'fit:', 'correspond:']):
                alignments.append(line.strip())
        return alignments

    def _extract_gaps(self, text: str) -> List[str]:
        """Extract gaps in meeting requirements."""
        gaps = []
        lines = text.split('\n')
        for line in lines:
            if any(marker in line.lower() for marker in ['gap:', 'missing:', 'lack:', 'needed:']):
                gaps.append(line.strip())
        return gaps

    def _extract_effectiveness(self, text: str) -> List[str]:
        """Extract effectiveness measures."""
        measures = []
        lines = text.split('\n')
        for line in lines:
            if any(marker in line.lower() for marker in ['effective:', 'measure:', 'metric:', 'impact:']):
                measures.append(line.strip())
        return measures

    def _extract_opposite_approaches(self, text: str) -> List[str]:
        """Extract opposite approaches from analysis."""
        opposites = []
        lines = text.split('\n')
        for line in lines:
            if any(marker in line.lower() for marker in ['opposite:', 'reverse:', 'contrary:', 'inverse:']):
                opposites.append(line.strip())
        return opposites

    def _format_critique_response(self, analysis: Dict, feedback: Dict, risks: Dict, 
                                critical_analyses: Optional[Dict[str, Dict]] = None) -> str:
        """Format the combined critique response with all critical thinking analyses."""
        response_parts = [
            "**Comprehensive Critical Analysis and Feedback**\n",
            "\n**Key Issues and Concerns:**\n"
        ]
        
        # Add identified issues
        if analysis["data"]["issues"]:
            response_parts.extend([f"- {issue}\n" for issue in analysis["data"]["issues"]])
            
        response_parts.append("\n**Constructive Feedback:**\n")
        if feedback["data"]["feedback"]:
            response_parts.append(feedback["data"]["feedback"])
            
        response_parts.append("\n**Risks and Mitigation Strategies:**\n")
        if risks["data"]["risks"]:
            response_parts.extend([f"- Risk: {risk}\n" for risk in risks["data"]["risks"]])
            
        if risks["data"]["mitigations"]:
            response_parts.append("\n**Recommended Mitigations:**\n")
            response_parts.extend([f"- {mitigation}\n" for mitigation in risks["data"]["mitigations"]])
        
        # Add results from additional critical thinking methods if available
        if critical_analyses:
            if "first_principles" in critical_analyses:
                response_parts.append("\n**First Principles Analysis:**\n")
                data = critical_analyses["first_principles"]["data"]
                if data["fundamental_elements"]:
                    response_parts.append("Fundamental Elements:\n")
                    response_parts.extend([f"- {element}\n" for element in data["fundamental_elements"]])
                
            if "critical_friend" in critical_analyses:
                response_parts.append("\n**Critical Friend Perspective:**\n")
                data = critical_analyses["critical_friend"]["data"]
                if data["probing_questions"]:
                    response_parts.append("Key Questions to Consider:\n")
                    response_parts.extend([f"- {question}\n" for question in data["probing_questions"]])
                
            if "devils_advocate" in critical_analyses:
                response_parts.append("\n**Devil's Advocate Analysis:**\n")
                data = critical_analyses["devils_advocate"]["data"]
                if data["counter_arguments"]:
                    response_parts.extend([f"- {arg}\n" for arg in data["counter_arguments"]])
                
            if "tenth_man" in critical_analyses:
                response_parts.append("\n**Tenth Man Perspective:**\n")
                data = critical_analyses["tenth_man"]["data"]
                if data["overlooked_risks"]:
                    response_parts.extend([f"- {risk}\n" for risk in data["overlooked_risks"]])
                
            if "objective_adherence" in critical_analyses:
                response_parts.append("\n**Objective Alignment Check:**\n")
                data = critical_analyses["objective_adherence"]["data"]
                if data["gaps"]:
                    response_parts.append("Gaps in Meeting Objectives:\n")
                    response_parts.extend([f"- {gap}\n" for gap in data["gaps"]])
                
            if "contrarian" in critical_analyses:
                response_parts.append("\n**Contrarian Perspective:**\n")
                data = critical_analyses["contrarian"]["data"]
                if data["opposite_approaches"]:
                    response_parts.append("Alternative Approaches to Consider:\n")
                    response_parts.extend([f"- {approach}\n" for approach in data["opposite_approaches"]])
        
        response_parts.append("\n**Suggestions for Improvement:**\n")
        if analysis["data"]["suggestions"]:
            response_parts.extend([f"- {suggestion}\n" for suggestion in analysis["data"]["suggestions"]])
            
        response_parts.append("\n**Final Recommendation:**\n")
        response_parts.append("Based on the comprehensive analysis above, here is the synthesized recommendation:")
        if analysis["data"]["suggestions"] and len(analysis["data"]["suggestions"]) > 0:
            response_parts.append(f"\n{analysis['data']['suggestions'][0]}")
            
        return "".join(response_parts)

    def generate_reply(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        sender: Optional[Agent] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any], None]:
        """Generate a reply with a focus on responding to team broadcasts."""
        try:
            if messages is None:
                messages = []

            # Check for the most recent broadcast
            recent_broadcasts = [
                msg for msg in messages if msg["role"] == "assistant" and msg.get("name") != self.name
            ]
            if recent_broadcasts:
                last_broadcast = recent_broadcasts[-1]
                return {
                    "role": "assistant",
                    "content": f"{self.name} responding to {last_broadcast['name']}: {last_broadcast['content']}"
                }

            # If no relevant broadcast, fall back to LLM-based generation
            ollama_messages = [{"role": "system", "content": self.system_message}]
            for msg in messages:
                if "content" in msg:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role not in ["system", "user", "assistant"]:
                        role = "user"
                    ollama_messages.append({"role": role, "content": content})

            model = self.llm_config.get("config_list", [{}])[0].get("model", "mistral")
            response = ollama.chat(model=model, messages=ollama_messages)

            content = self._extract_message_content(response)
            if content:
                return {
                    "role": "assistant",
                    "content": content
                }

            raise ValueError("Could not generate valid content")

        except Exception as e:
            logger.error(f"Error in generate_reply: {str(e)}")
            return {
                "role": "assistant",
                "content": "I apologize, but something went wrong. Please try again."
            }


    def _extract_content_for_analysis(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract content that needs critical analysis from messages."""
        content_indicators = [
            "analyze", "evaluate", "review", "assess",
            "proposal:", "plan:", "strategy:", "approach:"
        ]
        
        for msg in reversed(messages):
            content = msg.get("content", "")
            if content and isinstance(content, str):
                content_lower = content.lower()
                if any(indicator in content_lower for indicator in content_indicators):
                    return content
        
        return None

    def _format_critique_response(self, analysis: Dict, feedback: Dict, risks: Dict) -> str:
        """Format the combined critique response."""
        response_parts = [
            "**Critical Analysis and Feedback**\n",
            "\n**Key Issues and Concerns:**\n"
        ]
        
        # Add identified issues
        if analysis["data"]["issues"]:
            response_parts.extend([f"- {issue}\n" for issue in analysis["data"]["issues"]])
            
        response_parts.append("\n**Constructive Feedback:**\n")
        if feedback["data"]["feedback"]:
            response_parts.append(feedback["data"]["feedback"])
            
        response_parts.append("\n**Risks and Mitigation Strategies:**\n")
        if risks["data"]["risks"]:
            response_parts.extend([f"- Risk: {risk}\n" for risk in risks["data"]["risks"]])
            
        if risks["data"]["mitigations"]:
            response_parts.append("\n**Recommended Mitigations:**\n")
            response_parts.extend([f"- {mitigation}\n" for mitigation in risks["data"]["mitigations"]])
            
        response_parts.append("\n**Suggestions for Improvement:**\n")
        if analysis["data"]["suggestions"]:
            response_parts.extend([f"- {suggestion}\n" for suggestion in analysis["data"]["suggestions"]])
            
        return "".join(response_parts)