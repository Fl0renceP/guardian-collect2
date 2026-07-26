from services.claims_service import load_claims

claims = load_claims()

for claim in claims[:5]:
    print("----------------")
    print("Peril:", claim["peril"])
    print("Item:", claim["item_type"])
    print("Category:", claim["item_category"])
    print("Amount:", claim["amount"])
    print("Date:", claim["incident_at"])