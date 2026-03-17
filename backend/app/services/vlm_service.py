"""VLM Explanation Service — generate AI text explanation.

Responsibilities:
- consume evaluation metrics and structured feedback
- call a Vision-Language Model to produce natural language explanation
- return text that can be displayed to the learner

Note: VLM failure produces completed_with_warnings, not hard failure.
"""

from typing import Optional


async def generate_explanation(
    score: int,
    metrics: dict,
    feedback: dict,
) -> Optional[str]:
    """Generate AI explanation text from evaluation data.

    Returns explanation string or None if VLM is unavailable.
    """
    # TODO: integrate actual VLM (e.g. GPT-4V, LLaVA, etc.)
    return (
        f"Your practice scored {score}/100. "
        "This is a placeholder explanation. "
        "The AI model will provide detailed feedback once integrated."
    )
