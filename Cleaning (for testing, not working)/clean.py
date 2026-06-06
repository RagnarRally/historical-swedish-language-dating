import requests

with open("1524_GR.txt", encoding="utf-8") as f:
    ocr_text = f.read()

with open("clean.txt", encoding="utf-8") as f:
    clean = f.read()

prompt = f"""
Du är en OCR-korrigerare.

Uppgift:
Korrigera endast uppenbara OCR-fel.

Förbjudet:
- sammanfatta
- kommentera
- förklara
- översätta
- skriva på något annat språk
- skriva introduktioner
- skriva avslutningar

Behåll:
- radbrytningar
- styckeindelning
- äldre stavning
- skiljetecken

Returnera endast den korrigerade texten.

TEXT:

{ocr_text}
"""

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "Qwen3:8B",
        "prompt": prompt,
        "stream": False
    }
)

corrected = response.json()["response"]

with open("korrigerad.txt", "w", encoding="utf-8") as f:
    f.write(corrected)
