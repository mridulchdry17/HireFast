from app.models.db_models import JobPosting, Application
from app.repositories.job_repository import JobRepository
from typing import Optional, List, Dict, Any

class JobService:
    """Service for managing Job Postings and Applications (Logic Layer)."""
    
    @staticmethod
    def create_job(role: str, company_name: str = None, location: str = None, 
                   employment_type: str = None, job_description: str = None) -> JobPosting:
        """Logic for creating a job posting."""
        job = JobPosting(
            role=role,
            company_name=company_name,
            location=location,
            employment_type=employment_type,
            job_description=job_description
        )
        return JobRepository.save_job(job)

    @staticmethod
    def get_job_by_id(job_id: str) -> Optional[JobPosting]:
        """Fetch job via repository."""
        return JobRepository.find_job_by_id(job_id)

    @staticmethod
    def create_application(job_id: str, name: str, email: str, resume_path: str, 
                           resume_text: str = None) -> Application:
        """Logic for creating an application + AI analysis."""
        from app.services.ai_service import AIService
        from app.models.db_models import JobPosting
        
        # 1. Fetch Job Description
        job = JobPosting.query.get(job_id)
        jd_text = job.job_description if job else ""
        
        # 2. Get AI Score if resume text is provided
        score = 0.0
        if resume_text and jd_text:
            try:
                ai_service = AIService()
                analysis = ai_service.summarize_and_score_resume(resume_text, jd_text)
                score = analysis.get('score', 0.0)
            except Exception as e:
                print(f"Scoring error in service: {e}")
        
        # 3. Create and Save
        app = Application(
            job_id=job_id,
            candidate_name=name,
            candidate_email=email,
            resume_path=resume_path,
            similarity_score=score
        )
        return JobRepository.save_application(app)

    @staticmethod
    def get_all_applications() -> List[Dict[str, Any]]:
        """Fetch all applications from DB and format for frontend."""
        apps = JobRepository.get_all_applications()
        result = []
        for app in apps:
            result.append({
                'id': app.id,
                'name': app.candidate_name,
                'email': app.candidate_email,
                'role': app.job.role if app.job else 'Unknown',
                'similarity_score': app.similarity_score,
                'status': 'new', # Default for internal
                'resume_file_id': app.resume_file_id,
                'resume_path': app.resume_path,
                'created_at': app.created_at.isoformat()
            })
        return result
