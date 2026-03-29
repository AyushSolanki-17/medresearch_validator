from medresearch_validator.client import MedresearchValidatorEnv
from medresearch_validator.models import MedresearchValidatorAction

from medresearch_validator.utils.logger import get_logger
logger = get_logger(__name__)

def run_episode(client, obs):

    logger.info("Starting episode")
    logger.info("Observation:", obs)

    # Step 1: Analyze
    action1 = MedresearchValidatorAction(
        action_type="analyze",
        content="There is a contradiction between findings and hypothesis",
        confidence=0.8
    )
    result = client.step(action1)
    logger.info("Analyze Reward:", result.reward)


    # Step 2: Validate
    action2 = MedresearchValidatorAction(
        action_type="validate",
        content="No disease is present",
        confidence=0.9
    )
    result = client.step(action2)
    logger.info("Validate Reward:", result.reward)


    # Step 3: Refine
    action3 = MedresearchValidatorAction(
        action_type="refine",
        content="Final diagnosis: no disease",
        confidence=0.9
    )
    result = client.step(action3)
    logger.info("Final Diagnosis:", result)

with MedresearchValidatorEnv(base_url="http://localhost:8000").sync() as client:

    # Keep trying until we hit HARD task
    for _ in range(5):
        result = client.reset()
        obs = result.observation
        if result.observation.task_type == "hard":
            break


    run_episode(client,obs)