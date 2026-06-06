from collections import defaultdict
import pathlib as pl
import re

period_words = defaultdict(int)
period_sentences = defaultdict(int)

files = list(
    pl.Path("./dataset").glob("*.txt")
)

max_year = max(
    int(file.name[:4])
    for file in files
)

for file in files:

    year = int(file.name[:4])

    with open(file, encoding="utf-8") as f:
        text = f.read()

    # Normalisera whitespace
    text = re.sub(r"\s+", " ", text)

    words = len(text.split())
    sentences = len(
        re.findall(r"[.!?]+", text)
    )

    if year <= 1550:
        period = "1521–1550"
    elif year >= 2001:
        period = f"2001–{max_year}"
    else:
        start = (
            (year - 1551) // 50
        ) * 50 + 1551

        end = start + 49

        period = f"{start}–{end}"

    period_words[period] += words
    period_sentences[period] += sentences

print(
    f"{'Period':<15} "
    f"{'Ord':>12} "
    f"{'Meningar':>12}"
)

print("-" * 40)

for period in sorted(period_words):

    print(
        f"{period:<15} "
        f"{period_words[period]:>12,} "
        f"{period_sentences[period]:>12,}"
    )
