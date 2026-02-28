"""
Helper utility functions.
"""
import re
from typing import List, Optional
from app.config import Config

def split_text_for_post(text: str, max_length: int = 3000) -> List[str]:
    """
    Ensure the text is suitable for a single LinkedIn post while preserving formatting.
    LinkedIn standard posts allow up to 3000 characters.
    """
    # Remove Markdown bold/italics
    text = re.sub(r'\*\*|\*', '', text)
    
    # Trim to max_length to ensure it fits in one post if necessary
    if len(text) > max_length:
        text = text[:max_length-3] + "..."
    
    return [text]

def get_form_link_for_role(role: str) -> str:
    """
    Get the appropriate application form link for a role.
    
    Args:
        role: Job role
        
    Returns:
        Form link URL
    """
    role_lower = role.lower()
    for keyword, link in Config.ROLE_FORM_LINKS.items():
        if keyword in role_lower:
            return link
    return Config.GENERIC_FORM_LINK

def extract_drive_file_id(drive_url: str) -> Optional[str]:
    """
    Extract file ID from Google Drive share link.
    
    Args:
        drive_url: Google Drive share URL
        
    Returns:
        File ID or None if not found
    """
    if not drive_url:
        print("Empty drive URL provided")
        return None
        
    print(f"Extracting file ID from: {drive_url}")
    
    # Try different patterns
    patterns = [
        r'/d/([\w-]+)',  # Standard share link
        r'id=([\w-]+)',  # ID parameter
        r'/file/d/([\w-]+)',  # Direct file link
        r'/drive/folders/([\w-]+)',  # Folder link (for debugging)
        r'([\w-]{25,})'  # Generic long ID pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, drive_url)
        if match:
            file_id = match.group(1)
            print(f"Found file ID: {file_id} using pattern: {pattern}")
            return file_id
    
    print(f"No file ID found in URL: {drive_url}")
    return None

def pick_best_resume(applicants: List[dict]) -> Optional[dict]:
    """
    Return the applicant with the highest similarity_score from a list.
    
    Args:
        applicants: List of applicants with similarity scores
        
    Returns:
        Best applicant or None if list is empty
    """
    if not applicants:
        return None
    best = max(applicants, key=lambda x: x.get('similarity_score', 0))
    return best
