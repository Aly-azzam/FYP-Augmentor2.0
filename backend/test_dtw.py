from app.utils.dtw_utils import compute_dtw


def main() -> None:
    seq1 = [[0, 0], [1, 1], [2, 2]]
    seq2 = [[0, 0], [2, 2]]

    result = compute_dtw(seq1, seq2)

    print("DTW test passed.")
    print(f"distance: {result['distance']}")
    print(f"normalized_distance: {result['normalized_distance']}")
    print(f"path: {result['path']}")


if __name__ == "__main__":
    main()
