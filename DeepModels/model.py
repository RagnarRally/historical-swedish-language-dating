from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pathlib as pl
from tqdm import tqdm
import time
import sys

from transformers.models.mllama.image_processing_mllama import _validate_mllama_preprocess_arguments
# ----------------------------
# GPU eller CPU
# ----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

# ----------------------------
# Ladda data
# ----------------------------

rows = []

for file in pl.Path("../dataset").glob("*.txt"):

    year = int(file.name[:4])

    with open(file, encoding="utf-8") as f:
        text = f.read()

    chunks = text.split(".")

    for chunk in chunks:

        chunk = chunk.strip()

        if len(chunk) > 30:

            rows.append({
                "year": year,
                "text": chunk
            })

texts = [row["text"] for row in rows]

labels = np.array(
    [row["year"] for row in rows],
    dtype=np.float32
)

# Normalisera år

labels = (labels - 1500) / 500

# ----------------------------
# Inbäddningsmodell
# ----------------------------
model_name = None

if len(sys.argv) > 1:
    model_name = sys.argv[1].lower()

if model_name != "emma":
    model_name = "kbbert"
    tokenizer = AutoTokenizer.from_pretrained( "KB/bert-base-swedish-cased")
    bert = AutoModel.from_pretrained( "KB/bert-base-swedish-cased").to(device)
else:
    embedder = SentenceTransformer( "google/embeddinggemma-300M", device=device)

embeddings_file_name = model_name + "_embeddings.pt"
best_model_file_name = model_name + "_best_model.pt"
embeddings_file = pl.Path(embeddings_file_name)
if embeddings_file.exists():
    embedding_time = 0
    embeddings = torch.load(embeddings_file_name) 
else:
    if model_name == "emma":
        start = time.perf_counter()
        embeddings = embedder.encode( texts, convert_to_tensor=True, show_progress_bar=True, batch_size=32)
        end = time.perf_counter()
        embedding_time = round(end - start)
        print("Tid inbäddningar:", embedding_time, "sekunder.")
    else:
        tokens = tokenizer( texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        # ----------------------------
        # SWE-BERT
        # ----------------------------  
        bert.eval()
        
        # ----------------------------
        # Create embeddings
        # ----------------------------
        start = time.perf_counter()
        embeddings = []
        with torch.no_grad():
        
            for i in tqdm( range(0, len(texts), 32), desc="Skapar inbäddningar"):
        
                batch_ids = (
                    tokens["input_ids"][i:i+32]
                    .to(device)
                )
        
                batch_mask = (
                    tokens["attention_mask"][i:i+32]
                    .to(device)
                )
        
                outputs = bert(
                    input_ids=batch_ids,
                    attention_mask=batch_mask
                )
        
                cls = outputs.last_hidden_state[:, 0, :]
        
                embeddings.append( cls.cpu())
        
        embeddings = torch.cat( embeddings, dim=0)
        end = time.perf_counter()
        embedding_time = round(end - start)
        print("Tid inbäddningar:", embedding_time)

print( "Inbäddningsform: ", embeddings.shape)

torch.save(embeddings, embeddings_file_name)

labels = torch.tensor( labels, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split( embeddings, labels, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=0.2, random_state=42)

dataset = TensorDataset( X_train.cpu(), y_train)

loader = DataLoader( dataset, batch_size=32, shuffle=True)

# ----------------------------
# Regressionmodel
# ----------------------------

embedding_dim = embeddings.shape[1]

class RegressionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear( embedding_dim, 256)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear( 256, 64)

        self.out = nn.Linear( 64, 1)

    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.relu(x)

        x = self.out(x)

        return x.squeeze(-1)

model = RegressionModel().to(device)

# ----------------------------
# Träning
# ----------------------------

optimizer = torch.optim.Adam( model.parameters(), lr=1e-3)

loss_fn = nn.MSELoss()

# ----------------------------
# Träna
# ----------------------------
val_dataset = TensorDataset( X_val.cpu(), y_val)

val_loader = DataLoader( val_dataset, batch_size=32)

best_val_loss = float("inf")
patience = 3
epochs_without_improvement = 0
start = time.perf_counter()
train_losses = []
val_losses = []
for epoch in range(100):

    model.train()

    total_loss = 0

    progress = tqdm(loader)

    for batch in progress:

        x, y = batch

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = loss_fn( pred, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress.set_description( f"Epoch {epoch+1}")

        progress.set_postfix( loss=loss.item())

    avg_train_loss = total_loss / len(loader)
    train_losses.append(avg_train_loss)

    print()
    print( "Genomsnittlig förlust:", total_loss / len(loader))
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            val_loss += loss.item()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(),best_model_file_name)
        print("Bästa modell sparad.")
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        print("Tidigt stop")
        break
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    print("Valideringsförlust:", val_loss)
end = time.perf_counter()
training_time = round(end - start)
print("Tid träning:", training_time, "sekunder.")
best_model_file = pl.Path(best_model_file_name)
if best_model_file.exists():
    model.load_state_dict(torch.load(best_model_file_name))

# ----------------------------
# Förutsägelse
# ----------------------------

test_texts = [
    "Med handel ok wandel med främmande folk mänges et språk.", #1678
    "Then andre engelen giöt vth sina skåål j haffuet, och thet wort som en dödh mandz blodh.", #1526
    "Det brukades sedan gammalt, att man var tre i båtlaget där på ön.", #1919
    "En swarter Morer eller Blåman.",#1654
    "Torpare och fattigt folk lega sig rum eller ställen vid ström-stranden, der de om nätterna stå och ösa Nors med håf.", #1755
    "Örnegått medh flogelds war.", #1573
    "Örråg kallas råg, smittad af en parasitsvamp Cladosporium herbarum, hvilken företrädesvis i våta år visar sig ss. svarta fläckar på rågkornen vid tiden för rågens mognad, orsakande skrumpna korn och till bakning svårbrukadt mjöl.", #1886
    "Jach gaff en liuflig rök ifrå mich, såsom Cinamer och kösteliga örter.", #1536
    "I köket hänger vitlöksflätor och i fönstret står det krukor med örter i och på bänken ligger det alltid färska tomater och avokado och paprika.", #2009
    "När föräldragården måst säljas hade systrarna tagit sig för med att väva.", #1956
    "När han var en växande gosse så var han qvick och munter." #1790
]

if model_name == "emma":
    test_embeddings = embedder.encode( test_texts, convert_to_tensor=True).to(device)

else:
    test_tokens = tokenizer( test_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

    with torch.no_grad():

        outputs = bert( input_ids=test_tokens["input_ids"].to(device), attention_mask=test_tokens["attention_mask"].to(device))

        test_embeddings = ( outputs.last_hidden_state[:, 0, :])

model.eval()

with torch.no_grad():

    pred = model(X_test.to(device))

pred = pred.cpu().numpy()
true = y_test.cpu().numpy()

# ----------------------------
# Avnormalisera årtal
# ----------------------------

pred_years = pred * 500 + 1500
true_years = true * 500 + 1500

model.eval()

with torch.no_grad():
    test_pred = model(test_embeddings.to(device))

test_pred = test_pred.cpu().numpy()

for text, year in zip(test_texts, test_pred):

    predicted_year = (year * 500 + 1500)

    print()
    print(text)
    print("Förutsade:", round(float(predicted_year))
    )

mae = mean_absolute_error(true_years,pred_years)

print("MAE:", round(mae, 2), "år")


rmse = root_mean_squared_error(true_years,pred_years)

print("RMSE:", round(rmse, 2), "år")
if embedding_time != 0:
    print("Tid träning och inbäddning:", embedding_time + training_time, "sekunder.")

plt.scatter(
    true_years,
    pred_years,
    alpha=0.4
)

plt.plot(
    [1500, 2000],
    [1500, 2000]
)

plt.xlabel(
    "Verkligt år"
)

plt.ylabel(
    "Förutsagt år"
)

plt.title(
    "Förutsagt mot verkligt årtal"
)

plt.figure()

plt.plot(
    train_losses,
    label="Träning"
)

plt.plot(
    val_losses,
    label="Validering"
)

plt.xlabel("Epok")
plt.ylabel("Förlust")
plt.title("Tränings- och valideringsförlust")
plt.legend()

plt.show()
