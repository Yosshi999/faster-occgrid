from argparse import ArgumentParser
import sqlite3
import pandas as pd

parser = ArgumentParser()
parser.add_argument("db")
parser.add_argument("--format", choices=["md", "tex"], default="md")
args = parser.parse_args()

conn = sqlite3.connect(args.db)
df = pd.read_sql_query("SELECT * FROM exp", conn)
if args.format == "md":
    print(df.to_markdown())
elif args.format == "tex":
    print(df.to_latex(float_format="%.2f", index=False))
else:
    raise NotImplementedError
conn.close()
