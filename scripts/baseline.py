from medresearch_validator.client import MedresearchValidatorEnv
from medresearch_validator.models import MedresearchValidatorAction

from medresearch_validator.utils.logger import get_logger
logger = get_logger(__name__)

def run_baseline():
    with MedresearchValidatorEnv(base_url="http://localhost:8000").sync() as client:
        result = client.reset()
        obs = result.observation

        # Simple rule-based agent
        if "no findings" in obs.image_findings.lower():
            content = "No disease detected"
        else:
            content = obs.image_findings

        # Analyze
        client.step(MedresearchValidatorAction(
            action_type="analyze",
            content="Analyzing findings",
            confidence=0.8
        ))

        # Validate
        client.step(MedresearchValidatorAction(
            action_type="validate",
            content=content,
            confidence=0.9
        ))

        # Refine
        client.step(MedresearchValidatorAction(
            action_type="refine",
            content=content,
            confidence=0.9
        ))

        # Get score
        score = client._client.post("/grader").json()
        logger.info("Baseline score",score)

if __name__ == "__main__":
    run_baseline()