"""
Data models for the MedResearch Validator environment.

Defines the structured Action and Observation types used by the agent
to interact with the environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MedresearchValidatorAction(Action):
    """
    Represents an action taken by the agent.

    Attributes:
        action_type (str): Type of reasoning step.
            Expected values: "analyze", "validate", "refine"
        content (str): Natural language reasoning or decision.
        confidence (float): Confidence score in range [0.0, 1.0]
    """

    action_type: str = Field(..., description="Type of action")
    content: str = Field(..., description="Agent reasoning or response")
    confidence: float = Field(..., description="Confidence score (0-1)")


class MedresearchValidatorObservation(Observation):
    """
    Represents the environment's observation returned to the agent.

    Attributes:
        image_findings (str): Structured description of medical image findings
        report_text (str): Radiology report text
        hypothesis (str): Current hypothesis being evaluated
        step_count (int): Number of steps taken in current episode
    """

    image_findings: str = Field(default="", description="Image findings summary")
    report_text: str = Field(default="", description="Radiology report text")
    hypothesis: str = Field(default="", description="Current hypothesis")
    step_count: int = Field(default=0, description="Step counter")
    task_type: str = Field(default="", description="Current task type")
