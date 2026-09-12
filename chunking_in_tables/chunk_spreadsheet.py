"""
1. Load the spreadsheet into a pandas DataFrame.
2. Group its rows into small batches (e.g. 8 rows at a time), always
   keeping whole rows together.
3. For EVERY batch, re-attach the table's title and column names, and
   turn each row into a natural-language sentence (embedding models are
   trained on language, not on comma-separated values, so a sentence-like
   row embeds far more usefully than a raw CSV line).
4. The result is a list of self-contained text chunks — each one could be
   handed to someone with zero other context and still make sense.
"""

from pathlib import Path

import pandas as pd


_SCRIPT_DIR = Path(__file__).resolve().parent
SPREADSHEET_PATH = _SCRIPT_DIR / "demo_course_sales.xlsx"
TABLE_TITLE = "Course Sales by Region and Quarter"
ROWS_PER_CHUNK = 8   # how many spreadsheet rows go into a single chunk


def row_to_sentence(row: pd.Series, columns: list) -> str:
    """Turn one row into "Region: APAC, Course: GenAI Bootcamp, ..." —
    a sentence-like line embeds better than a bare CSV row."""
    return ", ".join(f"{col}: {row[col]}" for col in columns)


def _row_batches(df: pd.DataFrame, rows_per_chunk: int):
    """Yield (start_index, batch_df) for consecutive, non-overlapping row
    batches. Slicing with .iloc always grabs whole rows, so a row's cells
    never get split across two chunks."""
    for start in range(0, len(df), rows_per_chunk):
        yield start, df.iloc[start : start + rows_per_chunk]


# ---------------------------------------------------------------------------
# Chunk the DataFrame, repeating the table title + column names in every
# chunk so each one is understandable on its own, with no other context.
# ---------------------------------------------------------------------------
def chunk_dataframe_by_rows(df: pd.DataFrame, table_title: str, rows_per_chunk: int) -> list:
    """Each batch of rows becomes one chunk of sentence-style lines."""
    columns = list(df.columns)
    chunks = []

    for start, batch in _row_batches(df, rows_per_chunk):
        row_lines = [
            f"Row {start + offset + 1} — {row_to_sentence(row, columns)}"
            for offset, (_, row) in enumerate(batch.iterrows())
        ]
        chunk_text = (
            f"Table: {table_title}\n"
            f"Columns: {', '.join(columns)}\n\n"
            + "\n".join(row_lines)
        )
        chunks.append(chunk_text)

    return chunks


# ---------------------------------------------------------------------------
# BONUS — an alternative representation: a markdown table per chunk.
# Same batching as above, but keeps the row/column grid visually intact,
# which can help when a downstream LLM needs to reason across rows at once
# (e.g. "which region had the highest growth?"). There's no universally
# correct choice — try both against your own retrieval questions.
# ---------------------------------------------------------------------------
def chunk_dataframe_as_markdown(df: pd.DataFrame, table_title: str, rows_per_chunk: int) -> list:
    columns = list(df.columns)
    chunks = []

    header_line = "| " + " | ".join(columns) + " |"
    separator_line = "| " + " | ".join(["---"] * len(columns)) + " |"

    for _, batch in _row_batches(df, rows_per_chunk):
        data_lines = [
            "| " + " | ".join(str(value) for value in row) + " |"
            for _, row in batch.iterrows()
        ]
        chunk_text = (
            f"Table: {table_title}\n\n"
            + "\n".join([header_line, separator_line, *data_lines])
        )
        chunks.append(chunk_text)

    return chunks


# ---------------------------------------------------------------------------
# Run the whole pipeline and print what's happening at each stage
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 70)
    print("STEP 1 — load the spreadsheet")
    print("=" * 70)
    if not SPREADSHEET_PATH.exists():
        raise FileNotFoundError(
            f"Expected demo spreadsheet not found: {SPREADSHEET_PATH}\n"
            "Place 'demo_course_sales.xlsx' next to this script, or point "
            "SPREADSHEET_PATH at your own spreadsheet, then re-run."
        )
    df = pd.read_excel(SPREADSHEET_PATH)
    print(f"Loaded '{SPREADSHEET_PATH.name}' -> {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows, as pandas shows them by default:\n")
    print(df.head())

    print("\n" + "=" * 70)
    print("STEP 2 — chunk it, sentence-style, "
          f"{ROWS_PER_CHUNK} rows per chunk")
    print("=" * 70)
    chunks = chunk_dataframe_by_rows(df, TABLE_TITLE, ROWS_PER_CHUNK)
    print(f"{df.shape[0]} rows -> {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks, start=1):
        print(f"--- CHUNK {i} of {len(chunks)} "
              f"({len(chunk)} characters) ---")
        print(chunk)
        print()

    print("=" * 70)
    print("STEP 3 (bonus) — the same first chunk, as a markdown table")
    print("=" * 70)
    markdown_chunks = chunk_dataframe_as_markdown(df, TABLE_TITLE, ROWS_PER_CHUNK)
    print(markdown_chunks[0])

    print("\n" + "=" * 70)
    print("WHY THIS MATTERS — try it yourself")
    print("=" * 70)
    print(
        "Read CHUNK 3 above on its own, pretending you never saw the rest\n"
        "of this output. You can still tell exactly what table it came\n"
        "from and what every number means — because the title and column\n"
        "names travel WITH the data, in every single chunk.\n\n"
        "Now compare that to what would happen if you chunked this file\n"
        "by raw character count instead (try replacing chunk_dataframe_by_rows\n"
        "with a plain text.split() every 300 characters) — you'll cut rows\n"
        "in half and separate data from its header. That's the exact bug\n"
        "this script exists to avoid."
    )
