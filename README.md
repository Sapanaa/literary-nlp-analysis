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
