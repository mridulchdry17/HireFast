"""
Resume processing service for candidate selection.
"""
import os
import tempfile
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from app.services.google_service import GoogleService
from app.config import Config

class ResumeService:
    """Service for resume processing and candidate selection."""
    
    def __init__(self):
        self.google_service = GoogleService()
    
    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes using LangChain's PyPDFLoader."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_file_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            text = " ".join([doc.page_content for doc in docs])
        finally:
            os.remove(tmp_file_path)
        return text
    
    def select_best_candidates(self, applicants: List[Dict], job_description: str) -> List[Dict]:
        """
        For each applicant, download and parse the PDF resume, compare to job description,
        and return the top 2 candidates (by cosine similarity).
        
        Args:
            applicants: List of applicant dictionaries
            job_description: Job description text
            
        Returns:
            List of top candidates with similarity scores
        """
        print(f"Starting candidate selection for {len(applicants)} applicants")
        resume_texts = []
        valid_applicants = []
        
        for i, applicant in enumerate(applicants):
            file_id = applicant['resume_file_id']
            print(f"Processing applicant {i+1}: {applicant['name']} (file ID: {file_id})")
            
            pdf_bytes = self.google_service.download_pdf_from_drive(file_id)
            if not pdf_bytes:
                print(f"Failed to download PDF for applicant: {applicant['name']}")
                continue
                
            try:
                print(f"Extracting text from PDF for applicant: {applicant['name']}")
                text = self.extract_text_from_pdf_bytes(pdf_bytes)
                print(f"Extracted text length: {len(text)} characters")
                
                if text.strip():
                    resume_texts.append(text)
                    valid_applicants.append(applicant)
                    print(f"Successfully processed applicant: {applicant['name']}")
                else:
                    print(f"Empty text extracted for applicant: {applicant['name']}")
            except Exception as e:
                print(f"Error parsing PDF for applicant {applicant['name']}: {e}")
                continue
        
        print(f"Successfully processed {len(valid_applicants)} applicants")
        
        if not resume_texts:
            print("No valid resumes found")
            return []
            
        if len(resume_texts) == 1:
            print("Only one applicant found, returning it")
            return [{'applicant': valid_applicants[0], 'similarity_score': 1.0}]
        
        try:
            print("Vectorizing resumes and job description")
            # Vectorize resumes and job description
            vectorizer = TfidfVectorizer().fit(resume_texts + [job_description])
            resume_vecs = vectorizer.transform(resume_texts)
            jd_vec = vectorizer.transform([job_description])
            scores = cosine_similarity(resume_vecs, jd_vec).flatten()
            
            print(f"Similarity scores: {scores}")
            
            # Get top 2 indices (or all if less than 2)
            num_to_return = min(2, len(scores))
            top_indices = scores.argsort()[-num_to_return:][::-1]
            
            top_candidates = []
            for idx in top_indices:
                top_candidates.append({
                    'applicant': valid_applicants[idx],
                    'similarity_score': float(scores[idx])
                })
                print(f"Top candidate: {valid_applicants[idx]['name']} with score: {scores[idx]}")
                
            return top_candidates
            
        except Exception as e:
            print(f"Error in vectorization/similarity calculation: {e}")
            # Fallback: return all valid applicants with default scores
            fallback_candidates = []
            for applicant in valid_applicants:
                fallback_candidates.append({
                    'applicant': applicant,
                    'similarity_score': 0.5
                })
            return fallback_candidates[:2] if len(fallback_candidates) >= 2 else fallback_candidates
