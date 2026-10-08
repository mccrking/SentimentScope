# SentimentScope — Analyse de sentiment sur réseaux sociaux

Projet Python pour classifier tweets et avis clients en **positif, neutre, négatif**.
Interface Streamlit et traitement CSV en ligne de commande. Le projet utilise des
datasets importés : il ne collecte pas automatiquement de publications.

## Démarrer sous Windows / PowerShell

Depuis un clone du dépôt, créer l’environnement et installer les dépendances :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Lancer l’interface :

Double-cliquer sur **lancer.bat**, ou exécuter :

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Ouvrir l’adresse locale indiquée par Streamlit, généralement http://localhost:8501.
Choisir un moteur, analyser un texte ou importer un CSV, choisir la colonne texte,
puis lancer la classification. Les résultats peuvent être filtrés et exportés.

Le dataset proposé par défaut est **le fichier `data/tweets.csv`** :
2 243 textes anglais, colonne `content`. Il s’agit d’une copie du fichier fourni
pour ce projet ; sa provenance et son authenticité n’ont pas été vérifiées.
Vous pouvez sélectionner les exemples synthétiques séparément. Une barre de
progression indique l’avancement sur les gros fichiers.

## Moteurs

- **DistilBERT multilingue** : moteur par défaut, via Hugging Face Transformers et
  PyTorch. Le [modèle de lxyuan](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student)
  est distillé et produit directement trois classes. Il reste un modèle d’environ
  0,1 milliard de paramètres, pas un modèle miniature. Un téléchargement initial
  de plusieurs centaines de Mo est nécessaire ; les poids sont ensuite mis en cache.
  L’inférence s’effectue localement sur CPU. Le score affiché est celui de la classe
  prédite, pas une précision mesurée sur vos données.
- **VADER / NLTK** : baseline rapide pour l’anglais, sans réseau neuronal. Le
  lexique est téléchargé une fois. Polarité ≥ 0,05 : positif ; ≤ −0,05 : négatif ;
  sinon neutre. La polarité n’est pas une probabilité. Ne pas utiliser cette
  baseline pour évaluer des avis français.

Aucun remplacement automatique de moteur en cas d’erreur : l’erreur est affichée.
Les URL et mentions sont retirées ; négations, accents, emojis, hashtags et casse
sont conservés. Les textes sont tronqués à 512 tokens pour le transformer. Les
lignes vides, manquantes ou réduites à une URL/mention sont exportées avec le
statut `texte_vide`, sans inventer un sentiment neutre. Les doublons sont conservés.

## Format du dataset et commandes

Une ligne par texte. Les autres colonnes et les identifiants sont conservés comme
chaînes, sans conversion numérique. Le séparateur est détecté automatiquement ;
virgule, point-virgule et tabulation peuvent aussi être choisis explicitement.
L’encodage par défaut est UTF-8 ; l’interface accepte aussi Windows-1252 et Latin-1.
Les colonnes `text`, `content`, `tweet`, `texte`, `review` et `avis` sont reconnues
automatiquement. Exemple :

```csv
id,text
1,"Excellent service !"
2,"Le colis arrive demain."
3,"Produit cassé, très déçu."
```

```powershell
# Français / multilingue
.\.venv\Scripts\python.exe cli.py data/demo_fr.csv --output outputs/fr.csv

# Votre dataset complet : la colonne content est détectée automatiquement
.\.venv\Scripts\python.exe cli.py data/tweets.csv --output outputs/tweets_sentiments.csv

# Anglais : baseline rapide
.\.venv\Scripts\python.exe cli.py data/demo_en.csv --backend vader --output outputs/en.csv

# Votre fichier avec séparateur point-virgule
.\.venv\Scripts\python.exe cli.py avis.csv --column avis --sep ";" --output outputs/avis.csv

# Fichier Windows-1252
.\.venv\Scripts\python.exe cli.py avis.csv --encoding cp1252 --output outputs/avis.csv

# Vérifications du traitement, dont une vraie inférence VADER
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Colonnes ajoutées : `texte_nettoye`, `sentiment`, `score`, `statut`, `moteur`, et
scores des trois classes (transformer) ou `polarite` (VADER). Ces noms sont réservés
et un fichier les contenant déjà sera refusé pour éviter d’écraser des données.

Les fichiers `data/demo_fr.csv` et `data/demo_en.csv` sont **synthétiques et non
annotés**. Ils servent à démontrer le parcours, pas à publier des métriques.
Pour mesurer la qualité, utiliser un dataset annoté indépendant, par langue et
domaine, puis calculer une matrice de confusion et un F1 macro. Le sarcasme,
les sentiments mixtes et les variations linguistiques peuvent entraîner des erreurs.

## Vérification de cette version

Les dix tests automatisés passent, notamment la classification de tous les tweets
dans l’interface avec VADER, la lecture du fichier fourni,
la détection de séparateur, la conservation des identifiants, le parcours de l’interface VADER et
la suppression des résultats affichés après changement de moteur. Les deux
commandes CLI ont été exécutées avec les vrais moteurs. L’export français
conserve ses 12 lignes, dont deux sans texte exploitable. Sur les dix autres,
DistilBERT a prédit sept positifs et trois négatifs, sans neutre : ce résultat
illustre les limites du modèle, en particulier sur les phrases factuelles de ce
petit exemple. Aucune précision ni performance de généralisation n’est revendiquée.

Le champ `pseudo_label` du fichier fourni n’est pas utilisé pour prédire le
sentiment. Il est conservé dans l’export ; une comparaison avec ce champ mesure
uniquement un accord avec des pseudo-étiquettes, et non une précision certifiée.
Un projet fonctionnel ne signifie pas que 100 % des sentiments prédits sont justes.

Le traitement complet de `data/tweets.csv` avec DistilBERT a été exécuté et son
export vérifié : **2 243 lignes classifiées**, toutes les colonnes d’origine
conservées à l’identique, scores des trois classes normalisés. Les prédictions
sont 1 041 négatifs, 1 005 positifs et 197 neutres. Ce sont des sorties du modèle,
pas une répartition de référence. Résultats : `outputs/tweets_sentiments.csv`.
Le serveur local a également passé son contrôle de santé et répondu en HTTP 200.
