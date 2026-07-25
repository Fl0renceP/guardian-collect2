from services.claims_service import load_claims
from services.safety_score import calculate_safety_score


# Load claims from Cosmos DB
claims = load_claims()

print(f"Loaded claims: {len(claims)}")


# Take first claim as demo
claim = claims[1]


print("\nClaim selected:")
print("----------------")

for key, value in claim.items():
    print(f"{key}: {value}")


# Calculate score
result = calculate_safety_score(claim)


print("\nSafety Score")
print("----------------")
print("Score:", result["score"], "/100")
print("Risk Level:", result["risk_level"])

print("\nReasons:")
for reason in result["reasons"]:
    print("-", reason)