"""
Refinement Memory and Feedback Module

This module provides memory management for refinement passes and feedback-based refinement.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


class RefinementMemory:
    """
    Stores and manages refinement history for a user.
    Tracks passes, scores, and notes to enable learning from previous refinements.
    """
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
    
    def log_pass(self, original: str, refined: str, score: Optional[float] = None, notes: Optional[List[str]] = None):
        """
        Log a refinement pass to memory.
        
        Args:
            original: The original text before refinement
            refined: The refined text after processing
            score: Optional quality score for this pass
            notes: Optional list of notes about this pass
        """
        pass_entry = {
            "original": original,
            "refined": refined,
            "score": score,
            "notes": notes or [],
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(pass_entry)
    
    def last_output(self) -> Optional[str]:
        """Get the last refined output, if any."""
        if self.history:
            return self.history[-1].get("refined")
        return None
    
    def last_score(self) -> Optional[float]:
        """Get the last score, if any."""
        if self.history:
            return self.history[-1].get("score")
        return None
    
    def clear(self):
        """Clear all history."""
        self.history.clear()


def refine_with_feedback(
    api_key: str,
    original_text: str,
    heuristics: Optional[Dict[str, Any]] = None,
    memory: Optional[RefinementMemory] = None,
    flags: Optional[Dict[str, Any]] = None
) -> str:
    """
    Refine text using feedback from previous passes stored in memory.
    
    Args:
        api_key: OpenAI API key
        original_text: The text to refine
        heuristics: Optional heuristics to guide refinement
        memory: Optional RefinementMemory instance with previous passes
        flags: Optional flags to control refinement behavior
    
    Returns:
        The refined text
    """
    # Import here to avoid circular dependencies
    from language_model import OpenAIModel
    from settings import Settings
    
    # Create a temporary settings object with the API key
    temp_settings = Settings()
    temp_settings.openai_api_key = api_key
    
    # Initialize the model
    model = OpenAIModel(temp_settings)
    
    # Build context from memory if available
    context_parts = []
    if memory and memory.history:
        recent_passes = memory.history[-3:]  # Use last 3 passes for context
        context_parts.append("Previous refinement passes:")
        for i, pass_entry in enumerate(recent_passes, 1):
            score_info = f" (score: {pass_entry.get('score')})" if pass_entry.get('score') else ""
            context_parts.append(f"Pass {i}{score_info}: {pass_entry.get('refined', '')[:200]}...")
    
    # Build the prompt
    prompt_parts = []
    if context_parts:
        prompt_parts.append("\n".join(context_parts))
        prompt_parts.append("\n---\n")
    
    prompt_parts.append("Refine the following text:")
    prompt_parts.append(original_text)
    
    if heuristics:
        prompt_parts.append("\nHeuristics:")
        prompt_parts.append(str(heuristics))
    
    full_prompt = "\n".join(prompt_parts)
    
    # Use the model to refine
    try:
        refined_text = model.generate(full_prompt)
        return refined_text
    except Exception as e:
        # Fallback: return original text if refinement fails
        print(f"Warning: refine_with_feedback failed: {e}")
        return original_text


