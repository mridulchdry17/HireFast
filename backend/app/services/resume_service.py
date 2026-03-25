"""
Resume processing service for candidate selection.
"""
import os
import io
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.config import Config


class ResumeService:
    """Service for resume processing and candidate selection."""

    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes using pypdf."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = " ".join(page.extract_text() or "" for page in reader.pages)
            return text.strip()
        except Exception as e:
            print(f"PDF text extraction error: {e}")
            return ""

    def select_best_candidates(self, applicants: List[Dict], job_description: str) -> List[Dict]:
        """
        For each applicant, read their local resume PDF, compare to job description,
        and return the top 2 candidates by cosine similarity.

        Expects applicants to have: name, email, resume_path (local filename in UPLOAD_FOLDER)
        """
        print(f"Starting candidate selection for {len(applicants)} applicants")
        resume_texts = []
        valid_applicants = []

        upload_folder = getattr(Config, 'UPLOAD_FOLDER', 'uploads')

        for i, applicant in enumerate(applicants):
            resume_path = applicant.get('resume_path', '')
            if not resume_path:
                print(f"Applicant {applicant.get('name')} has no resume_path, skipping")
                continue

            full_path = os.path.join(upload_folder, resume_path)
            if not os.path.exists(full_path):
                print(f"Resume file not found on disk: {full_path}")
                continue

            try:
                with open(full_path, 'rb') as f:
                    pdf_bytes = f.read()

                print(f"Extracting text from PDF: {full_path}")
                text = self.extract_text_from_pdf_bytes(pdf_bytes)
                print(f"Extracted {len(text)} characters for {applicant.get('name')}")

                if text.strip():
                    resume_texts.append(text)
                    valid_applicants.append(applicant)
                else:
                    print(f"Empty text for {applicant.get('name')}, skipping")
            except Exception as e:
                print(f"Error reading resume for {applicant.get('name')}: {e}")
                continue

        print(f"Successfully processed {len(valid_applicants)} applicants")

        if not resume_texts:
            print("No valid resumes found")
            return []

        if len(resume_texts) == 1:
            return [{'applicant': valid_applicants[0], 'similarity_score': 1.0}]

        try:
            vectorizer = TfidfVectorizer().fit(resume_texts + [job_description])
            resume_vecs = vectorizer.transform(resume_texts)
            jd_vec = vectorizer.transform([job_description])
            scores = cosine_similarity(resume_vecs, jd_vec).flatten()

            print(f"Similarity scores: {scores}")

            num_to_return = min(2, len(scores))
            top_indices = scores.argsort()[-num_to_return:][::-1]

            top_candidates = []
            for idx in top_indices:
                top_candidates.append({
                    'applicant': valid_applicants[idx],
                    'similarity_score': float(scores[idx])
                })
                print(f"Top candidate: {valid_applicants[idx]['name']} score={scores[idx]:.3f}")

            return top_candidates

        except Exception as e:
            print(f"Error in vectorization: {e}")
            return [{'applicant': a, 'similarity_score': 0.5} for a in valid_applicants[:2]]


