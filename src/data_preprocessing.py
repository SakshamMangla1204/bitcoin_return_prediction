import pandas as pd


def parse_volume(value: str) -> float:
    value = str(value).replace(",", "").strip()
    if value in {"", "-"}:
        return 0.0
    if value.endswith("K"):
        return float(value[:-1]) * 1_000
    if value.endswith("M"):
        return float(value[:-1]) * 1_000_000
    if value.endswith("B"):
        return float(value[:-1]) * 1_000_000_000
    return float(value)


def main() -> None:
    df = pd.read_csv("data/raw/Bitcoin Historical Data.csv")
    print(df.head())
    print(df.shape)

    for col in ["Price", "Open", "High", "Low"]:
        df[col] = df[col].str.replace(",", "", regex=False).astype(float)

    df["Vol."] = df["Vol."].apply(parse_volume)
    df["Change %"] = df["Change %"].str.replace("%", "", regex=False).astype(float)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date", ascending=True).reset_index(drop=True)

    output_path = "data/processed/Bitcoin Historical Data Processed.csv"
    df.to_csv(output_path, index=False)

    print(df.shape)
    print(f"Saved cleaned data to: {output_path}")


if __name__ == "__main__":
    main()
