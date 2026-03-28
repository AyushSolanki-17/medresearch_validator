# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Medresearch Validator Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State


from medresearch_validator.models import (
    MedresearchValidatorAction,
    MedresearchValidatorObservation,
)


class MedresearchValidatorEnv(
    EnvClient[MedresearchValidatorAction, MedresearchValidatorObservation, State]
):
    """
    Client for the Medresearch Validator Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MedresearchValidatorEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(MedresearchValidatorAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MedresearchValidatorEnv.from_docker_image("medresearch_validator-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(MedresearchValidatorAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MedresearchValidatorAction) -> Dict:
        """
        Convert MedresearchValidatorAction to JSON payload for step message.

        Args:
            action: MedresearchValidatorAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[MedresearchValidatorObservation]:
        """
        Parse server response into StepResult[MedresearchValidatorObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MedresearchValidatorObservation
        """
        obs_data = payload.get("observation", {})
        observation = MedresearchValidatorObservation(
            image_findings=obs_data.get("image_findings", ""),
            report_text=obs_data.get("report_text", ""),
            hypothesis=obs_data.get("hypothesis", ""),
            step_count=obs_data.get("step_count", 0),
            task_type=obs_data.get("task_type", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
