from medresearch_validator.client import MedresearchValidatorEnv
from medresearch_validator.models import MedresearchValidatorAction

with MedresearchValidatorEnv(base_url="http://localhost:8000").sync() as client:
    print("Starting MedresearchValidatorAction...")
    result = client.reset()
    print(result.observation)

    print("Fetching MedresearchValidatorAction...")
    print("CLIENT MODEL:", MedresearchValidatorAction)


    action = MedresearchValidatorAction(
        action_type="validate",
        content="This suggests pneumonia",
        confidence=0.9
    )

    print(action.model_dump())  # 👈 check payload

    result = client.step(action)
    print(result.observation)

    print("Ending MedresearchValidatorAction...")