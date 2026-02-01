# Literary Text Analysis using NLP

This project applies Natural Language Processing techniques to analyze
classic literary texts, uncovering dominant themes, linguistic patterns,
and narrative structure.

## Dataset
- Moby-Dick by Herman Melville (public domain)

## Current Progress
- Project setup
- Data ingestion and HTML text extraction

## Logging
The project includes structured logging to track data ingestion and
processing steps. Logs are written to `logs/app.log`.

### spaCy Model
This project uses the `en_core_web_sm` spaCy model for tokenization,
lemmatization, and linguistic preprocessing.

### Large Text Handling
Long documents are processed in chunks and with disabled parser/NER
components to ensure memory-efficient preprocessing.

## Exploratory Data Analysis
The project includes text-level EDA such as vocabulary statistics,
lexical diversity, frequent word analysis, and Zipf’s Law visualization.

## Visualizations
All plots are automatically saved to the `results/images/` directory,
including word frequency charts and Zipf’s Law distributions.

## Step: 4 N-gram & Phrase Analysis
**Goal**: Identify frequent word combinations (phrases) that capture context and meaning better than single tokens.

## STEP 5 — TF-IDF Feature Engineering
**Goal**: Convert your cleaned text into numerical features.

The project represents chapters using TF-IDF features to identify
terms and phrases that are most distinctive across the narrative.



## REPORT
I used structural pattern matching to extract the narrative content of a public-domain text, removing metadata and boilerplate before feature engineering.

The cleaned corpus contains approximately 111k tokens with a vocabulary size of ~13k, indicating rich linguistic variety. The lexical diversity is consistent with long-form literary text, and the average word length reflects the descriptive nature of the domain

The word frequency distribution follows Zipf’s Law, indicating that the processed text exhibits statistical properties consistent with natural language.

TF-IDF analysis highlights core thematic concepts and character-specific terms, indicating strong narrative structure and chapter-level variation.

## Topic Modeling Results

Using NMF topic modeling on TF-IDF chapter representations, the following
latent themes were identified:

- **Topic 0 – Ahab and the White Whale**: obsession, leadership, pursuit
- **Topic 1 – Whaling and Cetology**: whale species, anatomy, oil
- **Topic 2 – Queequeg and Early Encounters**: early narrative and character relationships
- **Topic 3 – Archaic Language and Speech**: stylistic and biblical language
- **Topic 4 – Ship Authority and Ownership**: command structure and ownership
- **Topic 5 – Whale Hunt and Action**: active whaling scenes and crew dynamics


## Technologies
Python, spaCy, NLTK, scikit-learn, pandas, matplotlib