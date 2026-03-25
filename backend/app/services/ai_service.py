"""
AI and LLM service for job description generation.
"""
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from app.config import Config

class AIService:
    """Service for AI/LLM operations."""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY, 
            model='llama-3.3-70b-versatile'
        )
    
    def generate_job_description(self, role: str, company_name: str = None, location: str = None, 
                                 employment_type: str = None, additional_requirements: str = None) -> dict:
        """
        Generate a job description for the given role.
        
        Args:
            role: Job role
            company_name: Name of the company
            location: Job location
            employment_type: Type of employment
            additional_requirements: Specific requirements or details
            
        Returns:
            Dictionary with job description or error
        """
        if not role:
            return {'post_status': 'error', 'error': 'Role is required'}
        
        prompt_parts = [f"Write a concise, professional job description for the role of {role}"]
        if company_name:
            prompt_parts.append(f"at {company_name}")
        if location:
            prompt_parts.append(f"located in {location}")
        if employment_type:
            prompt_parts.append(f"as a {employment_type} position")
        
        main_prompt = " ".join(prompt_parts) + ", optimized for posting on LinkedIn.\n\n"
        
        if additional_requirements:
            main_prompt += f"Specific Requirements/Details to include:\n{additional_requirements}\n\n"
            
        template = f"""{main_prompt}
You are an HR and copywriting expert. Write a concise, professional job description optimized for LinkedIn.

Guidelines:
- Keep the total length under 1200 characters to allow space for the application link.
- Use clear, professional language with a compelling hook.
- Include key responsibilities, qualifications, and a call-to-action.
- Add relevant hashtags (e.g., #Hiring, #JobOpportunity) and emojis for engagement.
- Preserve newlines for readability.
- Do NOT start with phrases like 'Here is the job description'; start directly with the content.

Return the response as a plain text job post.

Role: {{role}}
"""

        prompt = PromptTemplate.from_template(template).format(role=role)
        
        try:
            jd = self.llm.invoke(prompt)
            return {'role': role, 'job_description': jd.content}
        except Exception as e:
            return {'post_status': 'error', 'error': f'JD generation failed: {str(e)}'}

    def summarize_and_score_resume(self, resume_text: str, job_description: str) -> dict:
        """
        Analyze a resume against a job description.
        
        Args:
            resume_text: Text extracted from resume
            job_description: Job description to match against
            
        Returns:
            Dictionary with score and short summary
        """
        template = """
        You are an expert technical recruiter. Analyze the following candidate's resume against the provided job description.
        
        Job Description:
        {jd}
        
        Resume Content:
        {resume}
        
        Tasks:
        1. Calculate a match percentage (0.0 to 1.0) based on skills, experience, and role requirements.
        2. Provide a 1-sentence summary of why they are or aren't a good fit.
        
        Return the result in this exact JSON format:
        {{"score": 0.85, "summary": "Candidate has strong Python experience but lacks the required Cloud certification."}}
        
        JSON Result:
        """
        
        prompt = template.format(jd=job_description, resume=resume_text)
        
        try:
            import json
            response = self.llm.invoke(prompt)
            # Try to find JSON in the response
            content = response.content.strip()
            if "{" in content and "}" in content:
                json_part = content[content.find("{"):content.rfind("}")+1]
                data = json.loads(json_part)
                return {
                    'score': float(data.get('score', 0.0)),
                    'summary': data.get('summary', 'Analysis completed.')
                }
            return {'score': 0.0, 'summary': 'Could not parse AI response.'}
        except Exception as e:
            print(f"AI Scoring error: {e}")
            return {'score': 0.0, 'summary': f'Error during analysis: {str(e)}'}
