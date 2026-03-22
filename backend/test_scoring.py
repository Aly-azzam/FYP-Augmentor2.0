from app.services.scoring_service import ScoringService


def main() -> None:
    scoring_service = ScoringService()
    result = scoring_service.compute_score(0.28)

    print("Scoring test passed.")
    print(f"score: {result['score']}")
    print(f"normalized_distance: {result['normalized_distance']}")


if __name__ == "__main__":
    main()
