"""
AI and LLM service for job description generation.
"""
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from app.config import Config

class AIService:
    """Service for AI/LLM operations."""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY, 
            model='openai/gpt-oss-120b'
        )
    
    def generate_job_description(self, role: str) -> dict:
        """
        Generate a job description for the given role.
        
        Args:
            role: Job role
            
        Returns:
            Dictionary with job description or error
        """
        if not role:
            return {'post_status': 'error', 'error': 'Role is required'}
        
        template = (
            "You are an HR and copywriting expert. Write a concise, professional job description for the role of {role}, optimized for posting on LinkedIn.\n\n"
            "Guidelines:\n"
            "- Keep the total length under 1200 characters to allow space for the application link.\n"
            "- Use clear, professional language with a compelling hook.\n"
            "- Include key responsibilities, qualifications, and a call-to-action.\n"
            "- Add relevant hashtags (e.g., #Hiring, #JobOpportunity) and emojis for engagement.\n"
            "- Avoid bullet points or excessive formatting; use plain text with newlines for readability.\n"
            "- Do NOT start with phrases like 'Here is the job description' or any introductory statements; start directly with the content.\n\n"
            "Return the response as a plain text paragraph only.\n\n"
            "Role: {role}"
        )

        prompt = PromptTemplate.from_template(template).format(role=role)
        
        try:
            jd = self.llm.invoke(prompt)
            return {'role': role, 'job_description': jd.content}
        except Exception as e:
            return {'post_status': 'error', 'error': f'JD generation failed: {str(e)}'}
