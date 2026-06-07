import pathlib as pl
import requests

# ----------------------------
# Läs filer
# ----------------------------

with open("1524_GR.txt", encoding="utf-8") as f:
    ocr_text = f.read()

with open("clean.txt", encoding="utf-8") as f:
    clean = f.read()

# ----------------------------
# Dela OCR-text i stycken
# ----------------------------

paragraphs = ocr_text.split("\n\n")

# ----------------------------
# Bygg block
# ----------------------------

TARGET_SIZE = 3000

blocks = []
current = ""

for paragraph in paragraphs:

    if len(current) + len(paragraph) > TARGET_SIZE:

        blocks.append(current.strip())
        current = paragraph

    else:

        if current:
            current += "\n\n"

        current += paragraph

if current:
    blocks.append(current.strip())

print(f"Blocks: {len(blocks)}")

# ----------------------------
# OCR-korrigering
# ----------------------------

corrected_chunks = []

for i, block in enumerate(blocks, start=1):

    print(f"Processing block {i}/{len(blocks)}")

    prompt = f"""
Du är en OCR-korrigerare.

Nedan följer ett exempel på korrekt transkriberad text.

EXEMPEL:

{clean}

Uppgift:

Korrigera endast OCR-fel.

Förbjudet:
- modernisera språket
- sammanfatta
- kommentera
- förklara
- översätta
- skriva på annat språk

Behåll:
- äldre stavning
- radbrytningar
- styckeindelning
- interpunktion

Returnera endast den korrigerade texten.

TEXT:

{block}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:4b",
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    corrected = response.json()["response"]

    corrected_chunks.append(corrected)

# ----------------------------
# Skriv resultat
# ----------------------------

with open("korrigerad.txt", "w", encoding="utf-8") as f:

    f.write("\n\n".join(corrected_chunks))

print("Done.")