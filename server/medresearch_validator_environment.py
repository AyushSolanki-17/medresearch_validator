"""
MedResearch Validator Environment.

Simulates a medical reasoning workflow where an agent evaluates hypotheses
based on image findings and reports using real dataset labels.

Key Features:
- Multi-task environment (easy, medium, hard)
- Deterministic reward + grading
- Real NIH dataset integration
"""

from typing import Optional, Any, List
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from medresearch_validator.models import (
    MedresearchValidatorAction,
    MedresearchValidatorObservation,
)

from server.utils import NIHDataLoader


class MedresearchValidatorEnvironment(Environment):
    """
    Medical reasoning environment for hypothesis validation.

    Tasks:
        - Easy: Validate diagnosis
        - Medium: Analyze + validate
        - Hard: Multi-step reasoning + refinement

    Reward Strategy:
        +0.3 → correct validation
        +0.2 → useful analysis
        +0.5 → refinement (task completion)
        -0.1 → incorrect reasoning
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    ACTION_TYPE = MedresearchValidatorAction
    OBSERVATION_TYPE = MedresearchValidatorObservation

    def __init__(self):
        """Initialize environment."""
        super().__init__()

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False

        self._loader = NIHDataLoader()

        self._task: str = "easy"
        self._ground_truth: List[str] = []
        self._history: List[MedresearchValidatorAction] = []

        # Current observation data
        self._image_findings: str = ""
        self._hypothesis: str = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MedresearchValidatorObservation:
        """
        Reset environment for a new episode.

        Returns:
            MedresearchValidatorObservation: Initial state
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._history = []

        # Sample dataset
        sample = self._loader.sample()
        labels = sample.get("labels", [])

        if self._task == "hard":
            if not self._ground_truth:
                # Inject contradiction
                self._image_findings = "No findings"
                self._hypothesis = "Patient has pneumonia"

        # Handle labels safely
        if not labels:
            self._image_findings = "No findings"
            self._hypothesis = "No disease detected"
            self._ground_truth = []
        else:
            self._image_findings = ", ".join(labels)
            self._hypothesis = f"Patient has {labels[0]}"
            self._ground_truth = labels

        return MedresearchValidatorObservation(
            image_findings=self._image_findings,
            report_text="Auto-generated report based on findings",
            hypothesis=self._hypothesis,
            step_count=0,
            reward=0.0,
            done=False,
            task_type=self._task,
        )

    def step(
        self,
        action: MedresearchValidatorAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MedresearchValidatorObservation:
        """
        Execute one reasoning step.

        Args:
            action: Agent's action

        Returns:
            Updated observation
        """
        self._state.step_count += 1
        self._history.append(action)
        print("SERVER MODEL:", MedresearchValidatorAction)

        reward = 0.0
        done = False

        action_type = action.action_type.lower()
        content = action.content.lower()

        # Task-based reward
        if self._task == "easy":
            reward = self._easy_task(action_type, content)

        elif self._task == "medium":
            reward = self._medium_task(action_type, content)

        elif self._task == "hard":
            reward = self._hard_task(action_type, content)

            if action_type == "refine":
                done = True

        # Episode termination
        if self._state.step_count >= 3:
            done = True

        return MedresearchValidatorObservation(
            image_findings=self._image_findings,
            report_text="Auto-generated report based on findings",
            hypothesis=self._hypothesis,
            step_count=self._state.step_count,
            reward=reward,
            done=done,
            task_type=self._task,
        )

    @property
    def state(self) -> State:
        """Return current environment state."""
        return self._state

    def grade(self) -> float:
        """
        Evaluate entire episode.

        Scoring:
            + diagnosis correctness
            + evidence usage
            + confidence calibration
            - hallucination penalty
        """
        score = 0.0
        seen_labels = set()

        for action in self._history:
            content = action.content.lower()
            # Contradiction detection
            if not self._ground_truth:
                if "contradiction" in content or "inconsistent" in content:
                    score += 0.3

            # Correct diagnosis
            for label in self._ground_truth:
                if label.lower() in content and label not in seen_labels:
                    score += 0.4
                    seen_labels.add(label)

            # Evidence
            if "opacity" in content or "lung" in content:
                score += 0.2

            # Confidence
            if 0.6 <= action.confidence <= 0.95:
                score += 0.1

            # Hallucination penalty
            if "cancer" in content and "cancer" not in [l.lower() for l in self._ground_truth]:
                score -= 0.2

        return max(0.0, min(score, 1.0))

    # TASK LOGIC

    def _easy_task(self, action_type: str, content: str) -> float:
        """Simple validation task."""
        if action_type == "validate":
            if not self._ground_truth:
                return 0.3 if "no disease" in content or "normal" in content else -0.1
            return 0.3 if any(l.lower() in content for l in self._ground_truth) else -0.1
        return 0.0

    def _medium_task(self, action_type: str, content: str) -> float:
        """Analysis + validation."""
        if action_type == "analyze":
            return 0.2

        if action_type == "validate":
            if not self._ground_truth:
                return 0.3 if "no disease" in content else -0.1
            return 0.3 if any(l.lower() in content for l in self._ground_truth) else -0.1

        return 0.0

    def _hard_task(self, action_type: str, content: str) -> float:
        """
        Hard task: detect inconsistency + refine hypothesis.
        """

        # Detect contradiction
        if action_type == "analyze":
            if "contradiction" in content or "inconsistent" in content:
                return 0.3
            return 0.1

        # Validate reasoning
        if action_type == "validate":
            if not self._ground_truth:
                if "no disease" in content:
                    return 0.3
                return -0.1

        # Refine hypothesis
        if action_type == "refine":
            if "no disease" in content or "normal" in content:
                return 0.6
            return 0.2

        return 0.0