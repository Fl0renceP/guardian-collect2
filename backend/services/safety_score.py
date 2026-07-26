def calculate_safety_score(claim):

    score = 100
    reasons = []

    amount = claim.get("amount", 0)

    # Claim amount impact
    if amount > 300000:
        score -= 40
        reasons.append("Very high claim value")
    elif amount > 100000:
        score -= 30
        reasons.append("High claim value")
    elif amount > 50000:
        score -= 20
        reasons.append("Moderate claim value")
    elif amount > 10000:
        score -= 10
        reasons.append("Above average claim value")


    peril = claim.get("peril", "").lower()
    category = claim.get("item_category", "").lower()


    # Theft severity
    if "theft" in peril:
        score -= 15
        reasons.append("Theft-related claim")


    if "motor vehicle" in category:
        score -= 15
        reasons.append("Motor vehicle theft risk")

    vehicle = claim.get("vehicle_make", "").lower()

    luxury_vehicles = {
        "porsche",
        "bmw",
        "mercedes",
        "audi",
        "range rover",
    }

    if vehicle in luxury_vehicles:
        score -= 10
        reasons.append("Luxury vehicle involved")


    # Keep score between 0-100
    score = max(0, min(score, 100))


    # Risk classification
    if score >= 80:
        risk_level = "Low"
    elif score >= 60:
        risk_level = "Medium"
    else:
        risk_level = "High"


    return {
        "score": score,
        "risk_level": risk_level,
        "reasons": reasons
    }