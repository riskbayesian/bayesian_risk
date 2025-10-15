from __future__ import annotations

import sys
from pathlib import Path
import os
from typing import Optional
import duckdb
import contextlib
import pandas as pd
import atexit
import readline

def _resolve_default_csv_path() -> Path:
    """Resolve the default CSV path relative to the repo root.

    Assumes this file lives at open_world_risk/scripts/query_gpt_prior_csv.py.
    The CSV is at data/prior_GPT_data_09_14_2025/object_pair_safety_ratings_filtered.csv
    from the project root.
    """
    # .../open_world_risk/scripts/query_gpt_prior_csv.py -> project root (parents[2])
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root
        / "data"
        / "prior_GPT_data_09_14_2025"
        / "object_pair_safety_ratings_filtered.csv"
    )

class PriorCSVQuery:
    """Lightweight SQL interface over the prior CSV using DuckDB.

    - Creates an in-memory DuckDB connection
    - Exposes the CSV as table/view `ratings`
    - Provides helper methods for running SQL and inspecting the schema
    """

    def __init__(self, csv_path: Path | str):
        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        # In-memory DB
        self.con = duckdb.connect(database=":memory:")

        # Create a view over the CSV. read_csv_auto will infer schema.
        # Using options to be resilient to minor CSV issues.
        csv_literal = str(self.csv_path).replace("'", "''")  # escape single quotes
        self.con.execute(
            f"""
            CREATE OR REPLACE VIEW ratings AS
            SELECT * FROM read_csv_auto('{csv_literal}',
                                        IGNORE_ERRORS=true,
                                        SAMPLE_SIZE=-1,
                                        DATEFORMAT='%Y-%m-%d',
                                        TIMESTAMPFORMAT='%Y-%m-%d %H:%M:%S');
            """
        )

    def sql(self, query: str):
        """Execute SQL and return a pandas DataFrame."""
        return self.con.execute(query).df()

    def head(self, n: int = 5):
        """Return first n rows from ratings."""
        return self.sql(f"SELECT * FROM ratings LIMIT {int(n)}")

    def tables(self):
        """List tables/views available."""
        return self.sql("SHOW ALL TABLES")

    def describe(self, table: str = "ratings"):
        """Describe table columns and types."""
        return self.sql(
            f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
            """
        )

    def distinct(self, column: str, limit: Optional[int] = None):
        """Get distinct values for a column."""
        lim = f" LIMIT {int(limit)}" if limit else ""
        return self.sql(f"SELECT DISTINCT {column} FROM ratings ORDER BY 1{lim}")


@contextlib.contextmanager
def pd_option_context():
    with pd.option_context(
        "display.max_rows",
        50,
        "display.max_columns",
        120,
        "display.width",
        200,
    ):
        yield


def _print_banner(csv_path: Path) -> None:
    print(
        f"Loaded CSV into DuckDB view `ratings`: {csv_path}",
        file=sys.stderr,
    )
    print(
        "Examples (uncomment in main to run):\n"
        "  qi.head(5)\n"
        "  qi.describe()\n"
        "  qi.sql(\"SELECT COUNT(*) AS n FROM ratings\")\n"
        "  qi.sql(\"SELECT * FROM ratings WHERE safety_rating < 0.2 ORDER BY safety_rating LIMIT 10\")\n",
        file=sys.stderr,
    )


def build_interface(custom_csv_path: Optional[str] = None) -> PriorCSVQuery:
    csv_path = Path(custom_csv_path) if custom_csv_path else _resolve_default_csv_path()
    qi = PriorCSVQuery(csv_path)
    #_print_banner(qi.csv_path)
    return qi

def start_sql_repl(qi: PriorCSVQuery) -> None:
    _setup_readline_history()
    print("\nInteractive SQL session. Enter SQL (single-line runs without ';').")
    print("Commands: .help, .tables, .schema [table], .head [n], .history [n], .exit / exit / quit / \\q")
    print("EXAMPLE SQL Command:")
    print("ratings sql> SELECT object_a, object_b, safety_rating")
    print("...> FROM ratings")
    print("...> WHERE safety_rating < 0.2")
    print("...> ORDER BY safety_rating ASC")
    print("...> LIMIT 5;")
    buffer = []
    while True:
        try:
            prompt = "ratings sql> " if not buffer else "...> "
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {".exit", "exit", "quit", "\\q"}:
            break
        if line in {".help", "help"}:
            print(
                "Available commands:\n"
                "  .tables                 - list tables/views\n"
                "  .schema [table]        - show table schema (default: ratings)\n"
                "  .head [n]              - show first n rows (default: 5)\n"
                "  .history [n]           - show last n commands (default: 20)\n"
                "  .exit | exit | quit    - leave the REPL\n"
                "You can also run any SQL. End with ';' for multi-line queries;\n"
                "single-line statements run without ';'."
            )
            continue
        if line.startswith(".tables"):
            try:
                print(qi.tables())
            except Exception as exc:
                print(f"Error: {exc}")
            continue
        if line.startswith(".schema"):
            parts = line.split()
            table = parts[1] if len(parts) > 1 else "ratings"
            try:
                print(qi.describe(table))
            except Exception as exc:
                print(f"Error: {exc}")
            continue
        if line.startswith(".head"):
            parts = line.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            try:
                print(qi.head(n))
            except Exception as exc:
                print(f"Error: {exc}")
            continue

        # Single-line SQL without ';' executes immediately if buffer is empty
        if not buffer and not line.startswith('.') and not line.endswith(';'):
            try:
                readline.add_history(line)
                df = qi.sql(line)
                with pd_option_context():
                    print(df)
            except Exception as exc:
                print(f"Error: {exc}")
            continue

        buffer.append(line)
        if line.endswith(";"):
            query = "\n".join(buffer)
            buffer = []
            try:
                _consolidate_history_entry(query)
                df = qi.sql(query)
                with pd_option_context():
                    print(df)
            except Exception as exc:
                print(f"Error: {exc}")

        if line.startswith(".history"):
            parts = line.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 20
            _print_history(n)
            continue


def _history_path() -> Path:
    path = os.environ.get("OPEN_WORLD_RISK_SQL_HISTORY")
    if path:
        return Path(path).expanduser()
    return Path.home() / ".open_world_risk_sql_history"


def _setup_readline_history() -> None:
    try:
        readline.parse_and_bind("tab: complete")
        readline.parse_and_bind("set editing-mode emacs")
    except Exception:
        pass
    try:
        HISTORY_FILE = _history_path()
        if HISTORY_FILE.exists():
            readline.read_history_file(str(HISTORY_FILE))
        readline.set_history_length(1000)

        def _save_history() -> None:
            try:
                HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
                readline.write_history_file(str(HISTORY_FILE))
            except Exception:
                pass

        atexit.register(_save_history)
    except Exception:
        pass


def _print_history(n: int) -> None:
    try:
        total = readline.get_current_history_length()
        start = max(1, total - n + 1)
        for i in range(start, total + 1):
            item = readline.get_history_item(i)
            if item is not None:
                print(f"{i}: {item}")
    except Exception as exc:
        print(f"History unavailable: {exc}")


def _consolidate_history_entry(query: str) -> None:
    try:
        lines = query.splitlines()
        k = len(lines)
        # Remove the last k entries (the individual lines) if present
        for _ in range(k):
            if readline.get_current_history_length() > 0:
                readline.remove_history_item(readline.get_current_history_length() - 1)
        # Add the combined query once
        readline.add_history(query)
    except Exception:
        # Best-effort; ignore if readline doesn't support removal
        pass

def demo_head_and_count(qi: PriorCSVQuery) -> None:
    head_df = qi.head(5)
    print("First 5 rows:")
    print(head_df)

    count_df = qi.sql("SELECT COUNT(*) AS n_rows FROM ratings")
    n_rows = int(count_df["n_rows"].iat[0])
    print(f"Total rows: {n_rows}")


def main():
    # Optional: pass a custom CSV path as the first CLI arg
    custom = None
    repl = False
    for arg in sys.argv[1:]:
        if arg == "--repl":
            repl = True
        else:
            custom = arg
    qi = build_interface(custom) # qi: query interface

    # If interactive flag is provided, start a simple SQL REPL
    if repl:
        start_sql_repl(qi)
        sys.exit(0)

    # Sample query: show first 5 rows and total number of rows (non-interactive)
    demo_head_and_count(qi)

# ================================================================================================
# HOW TO RUN INTERACTIVELY
# python open_world_risk/scripts/query_gpt_prior_csv --repl
# ================================================================================================
if __name__ == "__main__":
    print("==================================================================")
    print("HOW TO RUN INTERACTIVELY")
    print("python open_world_risk/scripts/query_gpt_prior_csv --repl")
    print("==================================================================")
    
    main()

