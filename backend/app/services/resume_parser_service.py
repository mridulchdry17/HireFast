"""
Resume Parser Service for extracting text from PDF resumes.
"""
import os
import tempfile
from typing import Optional, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from app.services.google_service import GoogleService

class ResumeParserService:
    """Service for parsing and extracting text from resume PDFs."""
    
    def __init__(self):
        self.google_service = GoogleService()
    
    def extract_text_from_pdf_langchain(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using LangChain's PyPDFLoader.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            text = " ".join([doc.page_content for doc in docs])
            return text.strip()
            
        except Exception as e:
            raise Exception(f"PDF text extraction failed: {str(e)}")
    
    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes using LangChain's PyPDFLoader.
        
        Args:
            pdf_bytes: PDF file content as bytes
            
        Returns:
            Extracted text content
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_file_path = tmp_file.name
            
            try:
                # Extract text using the temporary file
                text = self.extract_text_from_pdf_langchain(tmp_file_path)
                return text
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                    
        except Exception as e:
            raise Exception(f"PDF bytes text extraction failed: {str(e)}")
    
    def parse_resume_from_drive(self, file_id: str) -> Dict[str, Any]:
        """
        Download and parse a resume from Google Drive.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            Dictionary with parsed resume data
        """
        try:
            # Download PDF from Google Drive
            pdf_bytes = self.google_service.download_pdf_from_drive(file_id)
            if not pdf_bytes:
                return {'error': 'Failed to download PDF from Google Drive'}
            
            # Extract text from PDF
            resume_text = self.extract_text_from_pdf_bytes(pdf_bytes)
            
            if not resume_text.strip():
                return {'error': 'No text content found in PDF'}
            
            # Parse resume structure (basic parsing)
            parsed_data = self.parse_resume_structure(resume_text)
            
            return {
                'success': True,
                'file_id': file_id,
                'raw_text': resume_text,
                'parsed_data': parsed_data,
                'text_length': len(resume_text)
            }
            
        except Exception as e:
            return {'error': f'Resume parsing failed: {str(e)}'}
    
    def parse_resume_structure(self, resume_text: str) -> Dict[str, Any]:
        """
        Parse resume text to extract structured information.
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            Dictionary with structured resume data
        """
        try:
            # Basic parsing - extract common sections
            lines = resume_text.split('\n')
            parsed = {
                'name': self.extract_name(lines),
                'email': self.extract_email(resume_text),
                'phone': self.extract_phone(resume_text),
                'skills': self.extract_skills(resume_text),
                'experience': self.extract_experience(resume_text),
                'education': self.extract_education(resume_text),
                'summary': self.extract_summary(resume_text)
            }
            
            return parsed
            
        except Exception as e:
            return {'error': f'Resume structure parsing failed: {str(e)}'}
    
    def extract_name(self, lines: list) -> Optional[str]:
        """Extract candidate name from resume lines."""
        try:
            # Usually the first non-empty line or first line with title case
            for line in lines[:5]:  # Check first 5 lines
                line = line.strip()
                if line and len(line.split()) >= 2 and line[0].isupper():
                    return line
            return None
        except:
            return None
    
    def extract_email(self, text: str) -> Optional[str]:
        """Extract email address from resume text."""
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        return matches[0] if matches else None
    
    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from resume text."""
        import re
        phone_patterns = [
            r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            r'\+?[0-9]{1,3}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}'
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        return None
    
    def extract_skills(self, text: str) -> list:
        """Extract skills from resume text."""
        # Common technical skills keywords
        skill_keywords = [
            'python', 'javascript', 'java', 'react', 'node.js', 'sql', 'mongodb',
            'aws', 'docker', 'kubernetes', 'git', 'html', 'css', 'typescript',
            'angular', 'vue', 'django', 'flask', 'spring', 'c++', 'c#', 'php',
            'machine learning', 'ai', 'data science', 'analytics', 'tableau',
            'power bi', 'excel', 'project management', 'agile', 'scrum'
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        for skill in skill_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        return list(set(found_skills))  # Remove duplicates
    
    def extract_experience(self, text: str) -> list:
        """Extract work experience from resume text."""
        # Simple extraction - look for common experience indicators
        experience_keywords = ['experience', 'work history', 'employment', 'career']
        experience_sections = []
        
        lines = text.split('\n')
        in_experience_section = False
        
        for line in lines:
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in experience_keywords):
                in_experience_section = True
                continue
            
            if in_experience_section and line.strip():
                # Look for job titles or company names
                if any(word in line_lower for word in ['engineer', 'developer', 'manager', 'analyst', 'consultant']):
                    experience_sections.append(line.strip())
        
        return experience_sections[:5]  # Return top 5 experiences
    
    def extract_education(self, text: str) -> list:
        """Extract education information from resume text."""
        education_keywords = ['education', 'degree', 'university', 'college', 'bachelor', 'master', 'phd']
        education_sections = []
        
        lines = text.split('\n')
        in_education_section = False
        
        for line in lines:
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in education_keywords):
                in_education_section = True
                continue
            
            if in_education_section and line.strip():
                if any(word in line_lower for word in ['university', 'college', 'institute', 'school']):
                    education_sections.append(line.strip())
        
        return education_sections[:3]  # Return top 3 education entries
    
    def extract_summary(self, text: str) -> Optional[str]:
        """Extract professional summary or objective from resume text."""
        summary_keywords = ['summary', 'objective', 'profile', 'about']
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in summary_keywords):
                # Get the next few lines as summary
                summary_lines = []
                for j in range(i + 1, min(i + 4, len(lines))):
                    if lines[j].strip():
                        summary_lines.append(lines[j].strip())
                if summary_lines:
                    return ' '.join(summary_lines)
        
        return None
    
    def get_resume_summary_for_interview(self, resume_text: str) -> str:
        """
        Generate a concise summary of the resume for AI interview context.
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            Formatted summary for interview context
        """
        try:
            parsed = self.parse_resume_structure(resume_text)
            
            summary_parts = []
            
            if parsed.get('name'):
                summary_parts.append(f"Name: {parsed['name']}")
            
            if parsed.get('summary'):
                summary_parts.append(f"Professional Summary: {parsed['summary']}")
            
            if parsed.get('skills'):
                skills_str = ', '.join(parsed['skills'][:10])  # Top 10 skills
                summary_parts.append(f"Key Skills: {skills_str}")
            
            if parsed.get('experience'):
                exp_str = '; '.join(parsed['experience'][:3])  # Top 3 experiences
                summary_parts.append(f"Recent Experience: {exp_str}")
            
            if parsed.get('education'):
                edu_str = '; '.join(parsed['education'][:2])  # Top 2 education
                summary_parts.append(f"Education: {edu_str}")
            
            return '\n\n'.join(summary_parts)
            
        except Exception as e:
            # Fallback to raw text if parsing fails
            return f"Resume Content:\n{resume_text[:1000]}..."  # First 1000 characters
