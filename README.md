#  Literary Text Analysis using NLP

This project applies **Natural Language Processing (NLP)** techniques to analyze classic literary texts, uncovering **dominant themes**, **linguistic patterns**, and **narrative structure** through statistical and machine-learning–based methods.

The analysis is implemented as a **modular, reproducible pipeline**, following best practices in data science and NLP.

---

##  Dataset

- **Moby-Dick** by *Herman Melville*  
- Source: **Project Gutenberg** (public domain)

---

##  Text Preprocessing

### spaCy Model
The project uses the `en_core_web_sm` spaCy model for:
- Tokenization  
- Lemmatization  
- Linguistic preprocessing  

### Large Text Handling
To efficiently process long documents:
- Text is processed in **chunks**
- spaCy parser and NER components are **disabled**
- Memory usage is kept within safe limits

Structural pattern matching is used to extract only the **narrative body** of the text, removing:
- Project Gutenberg metadata  
- Licensing text  
- Front matter  

---

##  Exploratory Data Analysis (EDA)

The project performs text-level EDA to validate preprocessing quality and linguistic structure, including:

- Token count and vocabulary size  
- Lexical diversity  
- Average word length  
- Word frequency analysis  
- Zipf’s Law validation  

These analyses confirm that the processed corpus exhibits statistical properties consistent with natural language.

---

##  Visualizations

All visual outputs are automatically saved to the `results/images/` directory, including:

- Word frequency distributions  
- Zipf’s Law plots  
- N-gram frequency charts  

This ensures reproducibility and separation of analysis from presentation.

---

##  Step 4 — N-gram & Phrase Analysis

**Goal:**  
Identify frequent multi-word expressions (bigrams and trigrams) that capture semantic context better than individual tokens.

This step highlights recurring phrases such as character names, locations, and thematic constructs.

---

##  Step 5 — TF-IDF Feature Engineering

**Goal:**  
Transform textual data into numerical representations suitable for machine learning.

- Chapters are treated as individual documents  
- TF-IDF features are used to identify terms and phrases that are **distinctive across the narrative**  
- Both unigrams and bigrams are included to improve interpretability  

---

##  Step 6 — Topic Modeling

Topic modeling is performed using **Non-negative Matrix Factorization (NMF)** on TF-IDF chapter representations.

### Identified Topics

- **Topic 0 – Ahab and the White Whale**  
  Obsession, leadership, pursuit  

- **Topic 1 – Whaling and Cetology**  
  Whale species, anatomy, oil  

- **Topic 2 – Queequeg and Early Encounters**  
  Early narrative and character relationships  

- **Topic 3 – Archaic Language and Speech**  
  Stylistic and biblical language  

- **Topic 4 – Ship Authority and Ownership**  
  Command structure and hierarchy  

- **Topic 5 – Whale Hunt and Action**  
  Active whaling scenes and crew dynamics  

These topics reflect both **thematic** and **stylistic** dimensions of the novel.

---

##  Topic Evolution Analysis

Topic strength is tracked across chapters to analyze how thematic emphasis changes throughout the narrative.

This enables:
- Identification of narrative phases  
- Analysis of character and theme prominence  
- Insight into structural progression of the text  

Results are saved as structured CSV files for further analysis or visualization.

---

##  Technologies Used

- **Python**
- **spaCy** – tokenization and lemmatization  
- **NLTK** – stopword removal  
- **scikit-learn** – TF-IDF feature extraction and NMF topic modeling  
- **NumPy & pandas** – data manipulation and analysis  
- **Matplotlib** – visualization  
- **Python logging** – structured pipeline logging  
- **Modular pipeline design** – reproducible and scalable architecture  

---

##  Logging & Reproducibility

The project includes structured logging to track:
- Data ingestion  
- Preprocessing steps  
- Model execution  


## Summary
This project demonstrates a **full NLP workflow**, from raw text ingestion to interpretable machine-learning outputs, combining:

- Linguistic preprocessing  
- Statistical validation  
- Feature engineering  
- Unsupervised learning  
- Human-interpretable analysis  

