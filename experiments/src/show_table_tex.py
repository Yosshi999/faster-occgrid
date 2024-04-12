from argparse import ArgumentParser
import sqlite3
import pandas as pd

parser = ArgumentParser()
parser.add_argument("db")
args = parser.parse_args()

conn = sqlite3.connect(args.db)
df = pd.read_sql_query("SELECT * FROM exp", conn)
print("psnr mean", df["psnr"].mean())
print("fps mean", df["fps"].mean())
if "mem" in df.columns:
    print("mem mean", df["mem"].mean() / 1024)
    df = df[["data", "psnr", "mem", "fps"]].set_index("data")
    df["mem"] = list(map(lambda x: f"{x / 1024:.1f}", df["mem"]))
else:
    df = df[["data", "psnr", "fps"]].set_index("data")
df = df.T

if "lego" in df.columns:  # NeRF-Synthetic
    print(df.to_latex(float_format="%.2f", index=True))
else:
    print(df["bicycle flowers garden stump treehill room counter kitchen bonsai".split()].to_latex(float_format="%.2f", index=True))
conn.close()
