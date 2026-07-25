from services.safety_score import calculate_safety_score


dummy_claim = {
    "incident": "Vehicle stolen outside shopping centre",
    "peril": "Vehicle Theft",
    "suburb": "STELLENBOSCH",
    "amount": 180000,
}


result = calculate_safety_score(dummy_claim)


print("==============================")
print("DISCOVERY INSURE SAFETY SCORE")
print("==============================")

print("\nClaim:")
print(dummy_claim["incident"])

print("\nScore:", result["score"], "/100")
print("Risk Level:", result["risk_level"])

print("\nReasons:")
for reason in result["reasons"]:
    print("-", reason)