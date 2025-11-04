"""
AI Interview Service for conducting automated interviews.
"""
import os
import secrets
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from groq import Groq
from app.models.ai_interview import (
    AIInterview, InterviewQuestion, InterviewAnswer, 
    InterviewEvaluation, InterviewSession
)
from app.config import Config

class AIInterviewService:
    """Service for managing AI-powered interviews."""
    
    def __init__(self):
        print(f"🔍 DEBUG: AIInterviewService initialized")
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        self.interviews = {}  # In-memory storage for demo (replace with database)
        self.conversation_histories = {}  # Track conversation context
        print(f"🔍 DEBUG: Groq client initialized with API key: {Config.GROQ_API_KEY[:10]}...")
    
    def generate_llm_response_groq(self, prompt: str, model: str = "llama-3.3-70b-versatile",
                                 temperature: float = 0.7) -> str:
        """Use Groq chat completions to get a response."""
        messages = [{"role": "user", "content": prompt}]
        resp = self.groq_client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
        )
        return resp.choices[0].message.content
    
    def generate_question_groq(self, resume_text: str, conversation_history: List[Dict[str, str]]) -> str:
        """Generate the next interview question based on resume and conversation history."""
        # Build history text
        history_text = "\n".join([
            f"Q: {h['question']}\nA: {h['answer']}" 
            for h in conversation_history
        ])
        
        prompt = f"""
You are an experienced technical interviewer conducting a professional interview. 
Use the candidate's resume and past conversation to ask one relevant, insightful question.

Resume Summary:
{resume_text}

Conversation so far:
{history_text}

Guidelines:
- Ask a follow-up question that builds on previous answers
- Focus on technical skills, experience, or behavioral aspects
- Keep questions clear and specific
- Avoid repeating previous questions
- Make questions relevant to the role and experience level

Ask the next question to continue the interview. Return only the question text.
"""
        return self.generate_llm_response_groq(prompt)
    
    def evaluate_answer_groq(self, question: str, answer: str) -> Dict[str, Any]:
        """Evaluate a candidate's answer and provide detailed feedback."""
        prompt = f"""
You are an expert interview evaluator. Analyze the candidate's answer and provide a comprehensive evaluation.

Question: {question}
Candidate's Answer: {answer}

Evaluate the answer on these criteria:
1. Technical Accuracy (0-10)
2. Clarity of Communication (0-10)
3. Depth of Understanding (0-10)
4. Problem-Solving Approach (0-10)
5. Relevance to Question (0-10)

Provide your response in this exact JSON format:
{{
    "overall_score": <number between 0-10>,
    "technical_accuracy": <number between 0-10>,
    "communication_clarity": <number between 0-10>,
    "depth_of_understanding": <number between 0-10>,
    "problem_solving": <number between 0-10>,
    "relevance": <number between 0-10>,
    "feedback": "<2-3 sentence overall feedback>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "suggestions": ["<suggestion 1>", "<suggestion 2>"]
}}

Be constructive and specific in your feedback.
"""
        
        try:
            response = self.generate_llm_response_groq(prompt)
            # Parse JSON response
            import json
            evaluation = json.loads(response)
            return evaluation
        except Exception as e:
            # Fallback evaluation if JSON parsing fails
            return {
                "overall_score": 5.0,
                "technical_accuracy": 5.0,
                "communication_clarity": 5.0,
                "depth_of_understanding": 5.0,
                "problem_solving": 5.0,
                "relevance": 5.0,
                "feedback": f"Answer received. Evaluation error: {str(e)}",
                "strengths": ["Provided an answer"],
                "weaknesses": ["Could not evaluate due to technical error"],
                "suggestions": ["Please try again"]
            }
    
    def create_interview_session(self, candidate_id: str, candidate_name: str, 
                               candidate_email: str, job_role: str, resume_text: str) -> AIInterview:
        """Create a new AI interview session."""
        interview_id = str(uuid.uuid4())
        interview_link = f"/ai-interviewer/{interview_id}"
        
        print(f"🔍 DEBUG: Creating interview session with ID: {interview_id}")
        print(f"🔍 DEBUG: Candidate: {candidate_name}, Role: {job_role}")
        
        interview = AIInterview(
            id=interview_id,
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            job_role=job_role,
            resume_text=resume_text,
            status='pending',
            started_at=None,
            completed_at=None,
            total_questions=5,  # Default 5 questions
            current_question=0,
            overall_score=None,
            interview_link=interview_link,
            created_at=datetime.now()
        )
        
        # Store in memory (replace with database)
        self.interviews[interview_id] = interview
        self.conversation_histories[interview_id] = []
        
        print(f"🔍 DEBUG: Interview created and stored. Total interviews: {len(self.interviews)}")
        
        return interview
    
    def create_quick_interview(self, candidate_name: str, job_role: str, resume_text: str = "") -> AIInterview:
        """Create a quick interview session for direct platform use."""
        interview_id = str(uuid.uuid4())
        interview_link = f"/ai-interviewer/{interview_id}"
        
        print(f"🔍 DEBUG: Creating quick interview with ID: {interview_id}")
        print(f"🔍 DEBUG: Quick candidate: {candidate_name}, Role: {job_role}")
        
        interview = AIInterview(
            id=interview_id,
            candidate_id=f"quick_{interview_id}",
            candidate_name=candidate_name,
            candidate_email="platform@hirefast.com",
            job_role=job_role,
            resume_text=resume_text or f"Interview for {job_role} position",
            status='pending',
            started_at=None,
            completed_at=None,
            total_questions=5,
            current_question=0,
            overall_score=None,
            interview_link=interview_link,
            created_at=datetime.now()
        )
        
        # Store in memory
        self.interviews[interview_id] = interview
        self.conversation_histories[interview_id] = []
        
        print(f"🔍 DEBUG: Quick interview created and stored. Total interviews: {len(self.interviews)}")
        
        return interview
    
    def start_interview(self, interview_id: str) -> Dict[str, Any]:
        """Start an interview session and generate the first question."""
        if interview_id not in self.interviews:
            return {'error': 'Interview session not found. Please create a new interview.'}
        
        interview = self.interviews[interview_id]
        if interview['status'] not in ['pending', 'in_progress']:
            return {'error': 'Interview already completed or cancelled'}
        
        # Update interview status
        interview['status'] = 'in_progress'
        interview['started_at'] = datetime.now()
        interview['current_question'] = 1
        
        # Generate first question
        conversation_history = self.conversation_histories[interview_id]
        question_text = self.generate_question_groq(interview['resume_text'], conversation_history)
        
        # Create question record
        question = InterviewQuestion(
            id=str(uuid.uuid4()),
            interview_id=interview_id,
            question_number=1,
            question_text=question_text,
            question_type='technical',  # Default type
            context=f"First question for {interview['job_role']} position",
            created_at=datetime.now()
        )
        
        return {
            'interview': interview,
            'question': question,
            'status': 'started'
        }
    
    def submit_answer(self, interview_id: str, question_id: str, 
                     answer_text: str, answer_audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Submit an answer and get evaluation."""
        if interview_id not in self.interviews:
            return {'error': 'Interview not found'}
        
        interview = self.interviews[interview_id]
        if interview['status'] != 'in_progress':
            return {'error': 'Interview not in progress'}
        
        # Create answer record
        answer = InterviewAnswer(
            id=str(uuid.uuid4()),
            interview_id=interview_id,
            question_id=question_id,
            question_number=interview['current_question'],
            answer_text=answer_text,
            answer_audio_path=answer_audio_path,
            answer_type='voice' if answer_audio_path else 'text',
            duration_seconds=None,  # Could be calculated from audio
            created_at=datetime.now()
        )
        
        # Get the question text (in a real app, this would come from database)
        # For now, we'll use a placeholder
        question_text = "Technical question"  # This should be retrieved from question_id
        
        # Evaluate the answer
        evaluation_data = self.evaluate_answer_groq(question_text, answer_text)
        
        # Create evaluation record
        evaluation = InterviewEvaluation(
            id=str(uuid.uuid4()),
            interview_id=interview_id,
            question_id=question_id,
            answer_id=answer['id'],
            score=evaluation_data['overall_score'],
            feedback=evaluation_data['feedback'],
            strengths=evaluation_data['strengths'],
            weaknesses=evaluation_data['weaknesses'],
            suggestions=evaluation_data['suggestions'],
            evaluation_criteria={
                'technical_accuracy': evaluation_data['technical_accuracy'],
                'communication_clarity': evaluation_data['communication_clarity'],
                'depth_of_understanding': evaluation_data['depth_of_understanding'],
                'problem_solving': evaluation_data['problem_solving'],
                'relevance': evaluation_data['relevance']
            },
            created_at=datetime.now()
        )
        
        # Update conversation history
        self.conversation_histories[interview_id].append({
            'question': question_text,
            'answer': answer_text,
            'feedback': evaluation_data['feedback']
        })
        
        # Check if interview is complete
        if interview['current_question'] >= interview['total_questions']:
            interview['status'] = 'completed'
            interview['completed_at'] = datetime.now()
            # Calculate overall score (average of all evaluations)
            interview['overall_score'] = evaluation_data['overall_score']  # Simplified for now
            
            return {
                'answer': answer,
                'evaluation': evaluation,
                'interview_complete': True,
                'overall_score': interview['overall_score']
            }
        else:
            # Generate next question
            interview['current_question'] += 1
            conversation_history = self.conversation_histories[interview_id]
            next_question_text = self.generate_question_groq(interview['resume_text'], conversation_history)
            
            next_question = InterviewQuestion(
                id=str(uuid.uuid4()),
                interview_id=interview_id,
                question_number=interview['current_question'],
                question_text=next_question_text,
                question_type='technical',
                context=f"Question {interview['current_question']} for {interview['job_role']} position",
                created_at=datetime.now()
            )
            
            return {
                'answer': answer,
                'evaluation': evaluation,
                'next_question': next_question,
                'interview_complete': False,
                'progress': f"{interview['current_question']}/{interview['total_questions']}"
            }
    
    def get_interview_status(self, interview_id: str) -> Dict[str, Any]:
        """Get the current status of an interview."""
        print(f"🔍 DEBUG: get_interview_status called with ID: {interview_id}")
        print(f"🔍 DEBUG: Available interviews: {list(self.interviews.keys())}")
        
        if interview_id not in self.interviews:
            print(f"🔍 DEBUG: Interview ID {interview_id} not found in interviews")
            return {'error': 'Interview not found'}
        
        interview = self.interviews[interview_id]
        print(f"🔍 DEBUG: Found interview: {interview.get('candidate_name', 'Unknown')}")
        print(f"🔍 DEBUG: Interview status: {interview.get('status', 'Unknown')}")
        
        return {
            'interview': interview,
            'conversation_history': self.conversation_histories.get(interview_id, [])
        }
    
    def get_all_interviews(self) -> List[AIInterview]:
        """Get all interview sessions (for admin dashboard)."""
        return list(self.interviews.values())
