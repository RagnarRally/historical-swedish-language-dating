from pandas.io.clipboards import to_clipboard
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor
from scipy.sparse import save_npz, load_npz
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import pathlib as pl
import numpy as np
import sys
import time

rows = []

for file in pl.Path("../dataset").glob("*.txt"):

    year = int(file.name[:4])

    with open(file, encoding="utf-8") as f:

        text = f.read()

    chunks = text.split(".")

    for chunk in chunks:

        chunk = chunk.strip()

        if len(chunk) > 30:

            rows.append({"year": year, "text": chunk})

texts = [row["text"] for row in rows]
labels = np.array(
    [row["year"] for row in rows],
    dtype=np.float32
)

# Gör om text till char n-gram-vektorer
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3,5)
)

vectorization_time = 0
matrix_file_name = "matrix.npz"
vectorizer_model_file = "tfidf_vectorizer.pkl"
vectorization_file = pl.Path(matrix_file_name)
if vectorization_file.exists():
    X = load_npz(matrix_file_name)
    vectorizer = joblib.load(vectorizer_model_file)
else:
    start = time.perf_counter()
    
    X = vectorizer.fit_transform(texts)
    
    end = time.perf_counter()
    
    vectorization_time = round(end - start)
    
    print("Vektoriseringstid: ", vectorization_time, "sekunder.")

    save_npz(matrix_file_name, X)
    joblib.dump( vectorizer,vectorizer_model_file)

# Träna regression
model_name = None

if len(sys.argv) > 1:
    model_name = sys.argv[1].lower()

if model_name == "linear":
    reg = LinearSVR(random_state=42, max_iter=500000)
elif model_name == "ridge":
    reg = Ridge(alpha=1.0)
else:
    reg = SGDRegressor(random_state=42, max_iter=500000)

# Normalisera år

labels = (labels - 1500) / 500

X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42
)

start = time.perf_counter()
reg.fit(X_train, y_train)
end = time.perf_counter()
training_time = round(end - start, 2)
print("Träningstid: ", training_time, "sekunder.")

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

# ----------------------------
# Förutsäga texter
# ----------------------------

X_custom = vectorizer.transform( test_texts)

custom_pred = reg.predict(
    X_custom
)

# ----------------------------
# Skriv ut resultat.
# ----------------------------

for text, year in zip( test_texts, custom_pred):
    print(text)
    print()

    predicted_year = ( year * 500 + 1500)

    print( "Förutsade:", round(predicted_year))

    print()

# ----------------------------
# Utvärdedr modell
# ----------------------------

pred = reg.predict( X_test)

pred_years = (pred * 500 + 1500)

true_years = (y_test * 500 + 1500)

mae = mean_absolute_error(
    true_years,
    pred_years
)

print(
    "MAE:",
    round(mae, 2),
    "år"
)

rmse = root_mean_squared_error(
    true_years,
    pred_years
)

print(
    "RMSE:",
    round(rmse, 2),
    "år"
)

total_time = training_time + vectorization_time
print("Total tid: ", total_time, "sekunder.")

# ----------------------------
# Scatterplot
# ----------------------------

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
    "Predicerat år"
)

plt.title(
    "Predicerat mot verkligt årtal"
)

plt.show()
