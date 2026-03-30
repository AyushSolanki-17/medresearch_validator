"""
MedResearch Validator Environment.

Simulates a medical reasoning workflow where an agent evaluates hypotheses
based on image findings and reports using real dataset labels.

Key Features:
- Multi-task environment (easy, medium, hard)
- Deterministic reward + grading
- Real NIH dataset integration
"""
import random
from typing import Optional, Any, List
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from medresearch_validator.models import (
    MedresearchValidatorAction,
    MedresearchValidatorObservation,
)

from medresearch_validator.utils import NIHDataLoader


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
        self._scenario_type: str = "normal"

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
        tasks = ["easy"] * 2 + ["medium"] * 3 + ["hard"] * 5
        self._task = random.choice(tasks)
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

            self._scenario_type = random.choice([
                "contradiction",
                "overconfidence",
                "ambiguity"
            ])
        else:
            self._scenario_type = "normal"

        # Handle labels safely
        if not labels:
            self._ground_truth = []

            if self._task == "hard":
                if self._scenario_type == "contradiction":
                    self._image_findings = "No findings"
                    self._hypothesis = "Patient has pneumonia"

                elif self._scenario_type == "overconfidence":
                    self._image_findings = "Mild opacity"
                    self._hypothesis = "Severe pneumonia"

                elif self._scenario_type == "ambiguity":
                    self._image_findings = "Possible infection"
                    self._hypothesis = "Patient has pneumonia"

            else:
                self._image_findings = "No findings"
                self._hypothesis = "No disease detected"

        else:
            self._ground_truth = labels

            if self._task == "hard":
                if self._scenario_type == "contradiction":
                    self._image_findings = ", ".join(labels)
                    self._hypothesis = "No disease detected"

                elif self._scenario_type == "overconfidence":
                    self._image_findings = "Mild " + labels[0]
                    self._hypothesis = "Severe " + labels[0]

                elif self._scenario_type == "ambiguity":
                    self._image_findings = "Possible " + labels[0]
                    self._hypothesis = f"Patient has {labels[0]}"

            else:
                self._image_findings = ", ".join(labels)
                self._hypothesis = f"Patient has {labels[0]}"

        return MedresearchValidatorObservation(
            image_findings=self._image_findings,
            report_text="Auto-generated report based on findings",
            hypothesis=self._hypothesis,
            step_count=0,
            reward=0.0,
            done=False,
            task_type=self._task,
            scenario_type=self._scenario_type,
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
            scenario_type=self._scenario_type,
        )

    @property
    def state(self) -> State:
        """Return current environment state."""
        return self._state

    def grade(self) -> float:
        """
        Advanced deterministic grader.

        Evaluates:
        - Correct diagnosis
        - Contradiction detection
        - Evidence usage
        - Confidence calibration
        - Penalizes repetition & hallucination
        """

        score = 0.0
        seen_labels = set()
        seen_phrases = set()

        for i, action in enumerate(self._history):
            content = action.content.lower().strip()

            # Penalize empty / weak responses
            if len(content) < 5:
                score -= 0.1
                continue

            # Penalize repetition
            if content in seen_phrases:
                score -= 0.1
                continue
            seen_phrases.add(content)

            # Contradiction detection (HARD task)
            if not self._ground_truth:
                if "contradiction" in content or "inconsistent" in content:
                    score += 0.3

            # Correct diagnosis (only once per label)
            for label in self._ground_truth:
                if label.lower() in content and label not in seen_labels:
                    score += 0.4
                    seen_labels.add(label)

            # Evidence usage
            if "opacity" in content or "lung" in content:
                score += 0.2

            # Confidence calibration
            if 0.6 <= action.confidence <= 0.95:
                score += 0.1

            # Hallucination penalty
            if "cancer" in content and "cancer" not in [l.lower() for l in self._ground_truth]:
                score -= 0.2

            # Reward progression (later steps more valuable)
            score += 0.05 * i

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
        Advanced hard task with multiple reasoning scenarios.
        """

        # --- CONTRADICTION ---
        if self._scenario_type == "contradiction":
            if action_type == "analyze":
                if "contradiction" in content or "inconsistent" in content:
                    return 0.3
                return 0.1

            if action_type == "refine":
                if not self._ground_truth:
                    if "no disease" in content:
                        return 0.6
                else:
                    if any(l.lower() in content for l in self._ground_truth):
                        return 0.6
                return 0.2

        # --- OVERCONFIDENCE ---
        if self._scenario_type == "overconfidence":
            if action_type == "analyze":
                if "mild" in content or "severity" in content:
                    return 0.3
                return 0.1

            if action_type == "refine":
                if "mild" in content:
                    return 0.6
                return 0.2

        # --- AMBIGUITY ---
        if self._scenario_type == "ambiguity":
            if action_type == "analyze":
                if "uncertain" in content or "possible" in content:
                    return 0.3
                return 0.1

            if action_type == "refine":
                if "uncertain" in content or "possible" in content:
                    return 0.6
                return 0.2

        return 0.0