"""
test_bq_loader.py — Sanity check for BQLoader
===============================================
Run from your project root:
    python test_bq_loader.py

What it checks:
  - BQLoader initializes correctly
  - run() loads all 3 tables and returns the right metadata
  - get_dataframes() returns the right keys, row counts, and columns
"""

from bq_loader import BQLoader


def main():
    print("=" * 60)
    print("Testing BQLoader")
    print("=" * 60)

    # --- init ---
    print("\n[1] Initializing BQLoader...")
    bq = BQLoader(config_path="data_pipeline/config/config.yaml")
    print("    OK")

    # --- run ---
    print("\n[2] Running acquisition...")
    metadata = bq.run()
    print(f"    timestamp  : {metadata['timestamp']}")
    print(f"    num_files  : {metadata['num_files']}  (expected 3)")
    print(f"    total_rows : {metadata['total_rows']}  (expected 509)")
    print(f"    files      : {metadata['files']}")
    print(f"    project    : {metadata['project']}")
    print(f"    dataset    : {metadata['dataset']}")

    # --- get_dataframes ---
    print("\n[3] Checking dataframes...")
    dataframes = bq.get_dataframes()

    expected = {
        "interpretations_raw": {
            "rows": 450,
            "cols": ["character_id", "character_name", "passage_id",
                     "book", "interpretation", "word_count"]
        },
        "passage_details_new": {
            "rows": 9,
            "cols": ["passage_id", "book_title", "passage_title",
                     "passage_text", "book_author", "chapter_number",
                     "num_interpretations"]
        },
        "user_details_new": {
            "rows": 50,
            "cols": ["Name", "Age", "Gender", "Profession", "Personality",
                     "Interest", "Reading_Intensity", "Experience_Level",
                     "Experience_Count", "Reading_Count", "Journey",
                     "Distribution_Category", "Style_1", "Style_2",
                     "Style_3", "Style_4"]
        },
    }

    for name, checks in expected.items():
        print(f"\n  {name}")
        df = dataframes[name]

        # row count
        row_ok = len(df) == checks["rows"]
        print(f"    rows    : {len(df)}  (expected {checks['rows']}) {'OK' if row_ok else 'MISMATCH'}")

        # columns
        missing = [c for c in checks["cols"] if c not in df.columns]
        extra   = [c for c in df.columns if c not in checks["cols"]]
        if missing:
            print(f"    missing cols : {missing}")
        if extra:
            print(f"    extra cols   : {extra}")
        if not missing and not extra:
            print(f"    columns : OK")

        # nulls
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if not nulls.empty:
            print(f"    nulls   : {nulls.to_dict()}")

        # first row preview
        print(f"    first row preview:")
        print(df.iloc[0].to_string(max_rows=5))

    print("\n" + "=" * 60)
    print("Done. If all tables show OK, BQLoader is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()