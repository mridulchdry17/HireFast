"""
AI Interview Service — uses SQLite DB for persistence (no more in-memory loss on restart).
Fixes:
  - Sessions stored in DB via AIInterviewSession / AIInterviewQuestion models
  - Resume loaded from local uploads/ folder (not Google Drive)
  - Real question text passed to evaluator (not hardcoded placeholder)
"""
import os
import io
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from groq import Groq
from app.config import Config


class AIInterviewService:
    """Service for managing AI-powered interviews using DB persistence."""

    def __init__(self):
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)

    # ─── LLM helpers ────────────────────────────────────────────────────────────

    def _llm(self, prompt: str, temperature: float = 0.7) -> str:
        resp = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=temperature,
        )
        return resp.choices[0].message.content

    def _generate_question(self, resume_text: str, history: List[Dict]) -> str:
        history_text = "\n".join(
            f"Q: {h['question']}\nA: {h['answer']}" for h in history
        )
        prompt = f"""You are an experienced technical interviewer conducting a professional interview.
Use the candidate's resume and past conversation to ask ONE relevant, insightful question.

Resume Summary:
{resume_text}

Conversation so far:
{history_text if history_text else "No previous questions yet — ask an opening question."}

Guidelines:
- Build on previous answers if any exist
- Focus on technical skills, experience, or behavioral aspects
- Keep questions clear and specific
- Avoid repeating previous questions

Return ONLY the question text, nothing else."""
        return self._llm(prompt)

    def _evaluate_answer(self, question: str, answer: str) -> Dict[str, Any]:
        prompt = f"""You are an expert interview evaluator. Analyze the candidate's answer.

Question: {question}
Candidate's Answer: {answer}

Evaluate on these criteria:
1. Technical Accuracy (0-10)
2. Clarity of Communication (0-10)
3. Depth of Understanding (0-10)
4. Problem-Solving Approach (0-10)
5. Relevance to Question (0-10)

Respond in this EXACT JSON format (no markdown, no extra text):
{{
    "overall_score": <number 0-10>,
    "technical_accuracy": <number 0-10>,
    "communication_clarity": <number 0-10>,
    "depth_of_understanding": <number 0-10>,
    "problem_solving": <number 0-10>,
    "relevance": <number 0-10>,
    "feedback": "<2-3 sentence overall feedback>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>"],
    "suggestions": ["<suggestion 1>"]
}}"""
        try:
            raw = self._llm(prompt, temperature=0.3)
            # Strip markdown code fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            print(f"Evaluation JSON parse error: {e}")
            return {
                "overall_score": 5.0,
                "technical_accuracy": 5.0,
                "communication_clarity": 5.0,
                "depth_of_understanding": 5.0,
                "problem_solving": 5.0,
                "relevance": 5.0,
                "feedback": "Answer received.",
                "strengths": ["Provided an answer"],
                "weaknesses": [],
                "suggestions": [],
            }

    # ─── Resume helpers ─────────────────────────────────────────────────────────

    def _load_resume_text(self, resume_path: str) -> str:
        """Read PDF from local uploads/ folder and extract text."""
        if not resume_path:
            return ""
        upload_folder = getattr(Config, 'UPLOAD_FOLDER', 'uploads')
        full_path = os.path.join(upload_folder, resume_path)
        if not os.path.exists(full_path):
            print(f"Resume not found on disk: {full_path}")
            return ""
        try:
            from pypdf import PdfReader
            with open(full_path, 'rb') as f:
                reader = PdfReader(io.BytesIO(f.read()))
            return " ".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""

    # ─── DB helpers ─────────────────────────────────────────────────────────────

    def _get_session(self, session_id: str):
        from app.models.db_models import AIInterviewSession
        return AIInterviewSession.query.get(session_id)

    def _get_history(self, session) -> List[Dict]:
        if not session.conversation_history:
            return []
        try:
            return json.loads(session.conversation_history)
        except Exception:
            return []

    def _save_history(self, session, history: List[Dict]):
        from app.models.db_models import db
        session.conversation_history = json.dumps(history)
        db.session.commit()

    # ─── Public API ─────────────────────────────────────────────────────────────

    def create_interview_session(self, application_id: str, candidate_name: str,
                                 candidate_email: str, job_role: str,
                                 resume_path: str = "") -> Dict:
        """Create a new persistent interview session from a DB application."""
        from app.models.db_models import db, AIInterviewSession

        resume_text = self._load_resume_text(resume_path)
        if not resume_text:
            resume_text = f"Candidate applying for {job_role}"

        session = AIInterviewSession(
            application_id=application_id,
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            job_role=job_role,
            resume_text=resume_text,
            status='pending',
            total_questions=5,
            current_question=0,
            conversation_history=json.dumps([]),
        )
        db.session.add(session)
        db.session.commit()

        print(f"Created interview session {session.id} for {candidate_name}")
        return self._session_to_dict(session)

    def start_interview(self, session_id: str) -> Dict:
        """Start the interview and generate the first question."""
        from app.models.db_models import db, AIInterviewQuestion

        session = self._get_session(session_id)
        if not session:
            return {'error': 'Interview session not found'}
        if session.status not in ('pending', 'in_progress'):
            return {'error': 'Interview already completed or cancelled'}

        history = self._get_history(session)
        question_text = self._generate_question(session.resume_text, history)

        session.status = 'in_progress'
        session.started_at = datetime.utcnow()
        session.current_question = 1

        q = AIInterviewQuestion(
            session_id=session_id,
            question_number=1,
            question_text=question_text,
        )
        db.session.add(q)
        db.session.commit()

        return {
            'session': self._session_to_dict(session),
            'question': {'id': q.id, 'question_number': 1, 'question_text': question_text},
            'status': 'started',
        }

    def submit_answer(self, session_id: str, question_id: str, answer_text: str) -> Dict:
        """Evaluate an answer, save the result, and generate the next question (or finish)."""
        from app.models.db_models import db, AIInterviewSession, AIInterviewQuestion

        session = self._get_session(session_id)
        if not session:
            return {'error': 'Interview session not found'}
        if session.status != 'in_progress':
            return {'error': 'Interview not in progress'}

        # Get the actual question text (fixes the hardcoded bug)
        question_obj = AIInterviewQuestion.query.get(question_id)
        question_text = question_obj.question_text if question_obj else "Interview question"

        # Evaluate the answer
        evaluation = self._evaluate_answer(question_text, answer_text)

        # Update conversation history
        history = self._get_history(session)
        history.append({
            'question': question_text,
            'answer': answer_text,
            'feedback': evaluation.get('feedback', ''),
        })
        self._save_history(session, history)

        # Check if interview is done
        if session.current_question >= session.total_questions:
            # Compute overall score as average of all answer scores stored in history
            all_scores = [h.get('score', evaluation['overall_score']) for h in history]
            overall = round(sum(all_scores) / len(all_scores), 2) if all_scores else evaluation['overall_score']

            session.status = 'completed'
            session.completed_at = datetime.utcnow()
            session.overall_score = overall

            # Update the linked application score
            if session.application_id:
                from app.models.db_models import Application
                app = Application.query.get(session.application_id)
                if app:
                    app.ai_interview_status = 'completed'
                    app.ai_interview_score = overall

            db.session.commit()
            return {
                'evaluation': evaluation,
                'interview_complete': True,
                'overall_score': overall,
            }
        else:
            # Store score in history for averaging later
            history[-1]['score'] = evaluation['overall_score']
            self._save_history(session, history)

            session.current_question += 1
            next_q_text = self._generate_question(session.resume_text, history)

            next_q = AIInterviewQuestion(
                session_id=session_id,
                question_number=session.current_question,
                question_text=next_q_text,
            )
            db.session.add(next_q)
            db.session.commit()

            return {
                'evaluation': evaluation,
                'next_question': {
                    'id': next_q.id,
                    'question_number': session.current_question,
                    'question_text': next_q_text,
                },
                'interview_complete': False,
                'progress': f"{session.current_question}/{session.total_questions}",
            }

    def get_interview_status(self, session_id: str) -> Dict:
        session = self._get_session(session_id)
        if not session:
            return {'error': 'Interview not found'}

        # Get the latest question so the page can display it after reload
        from app.models.db_models import AIInterviewQuestion
        last_q = AIInterviewQuestion.query.filter_by(session_id=session_id).order_by(
            AIInterviewQuestion.question_number.desc()
        ).first()

        return {
            'session': self._session_to_dict(session),
            'conversation_history': self._get_history(session),
            'last_question': {
                'id': last_q.id,
                'question_number': last_q.question_number,
                'question_text': last_q.question_text,
            } if last_q else None,
        }

    def get_session_by_application(self, application_id: str) -> Optional[Dict]:
        """Get the interview session for a specific application (if exists)."""
        from app.models.db_models import AIInterviewSession
        session = AIInterviewSession.query.filter_by(application_id=application_id).order_by(
            AIInterviewSession.created_at.desc()
        ).first()
        if session:
            return self._session_to_dict(session)
        return None

    def get_all_interviews(self) -> List[Dict]:
        from app.models.db_models import AIInterviewSession
        return [self._session_to_dict(s) for s in AIInterviewSession.query.order_by(
            AIInterviewSession.created_at.desc()
        ).all()]

    def _session_to_dict(self, session) -> Dict:
        return {
            'id': session.id,
            'application_id': session.application_id,
            'candidate_name': session.candidate_name,
            'candidate_email': session.candidate_email,
            'job_role': session.job_role,
            'status': session.status,
            'total_questions': session.total_questions,
            'current_question': session.current_question,
            'overall_score': session.overall_score,
            'interview_link': f"/ai-interviewer/{session.id}",
            'created_at': session.created_at.isoformat() if session.created_at else None,
            'started_at': session.started_at.isoformat() if session.started_at else None,
            'completed_at': session.completed_at.isoformat() if session.completed_at else None,
        }
