from app.models.db_models import db, JobPosting, Application
from typing import Optional, List

class JobRepository:
    """Repository for direct Job and Application database operations."""
    
    @staticmethod
    def save_job(job: JobPosting) -> JobPosting:
        db.session.add(job)
        db.session.commit()
        return job

    @staticmethod
    def find_job_by_id(job_id: str) -> Optional[JobPosting]:
        return JobPosting.query.get(job_id)

    @staticmethod
    def save_application(application: Application) -> Application:
        db.session.add(application)
        db.session.commit()
        return application

    @staticmethod
    def get_all_applications() -> List[Application]:
        return Application.query.order_by(Application.created_at.desc()).all()
