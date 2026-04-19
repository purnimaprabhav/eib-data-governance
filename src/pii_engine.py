import spacy
import pandas as pd
from spacy.language import Language

class PIIDetector:
    def __init__(self):
        """Initializes the spaCy NLP pipeline with custom PII patterns."""
        # Load the large model for better accuracy on names/locations
        self.nlp = spacy.load("en_core_web_md")
        
        # Define custom patterns for structured data (Regex)
        self.patterns = [
            # Email Pattern
            {"label": "EMAIL", "pattern": [{"TEXT": {"REGEX": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}}]},
            
            # Phone Number (Supports various formats)
            {"label": "PHONE", "pattern": [{"TEXT": {"REGEX": r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"}}]},
            
            # IBAN (General European pattern)
            {"label": "IBAN", "pattern": [{"TEXT": {"REGEX": r"[A-Z]{2}\d{2}[A-Z0-9]{12,30}"}}]}
        ]
        
        # Add EntityRuler to the pipeline BEFORE the standard NER
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(self.patterns)

    def scan_column(self, series):
        """
        Scans a pandas series for PII. 
        Returns a unique list of detected entity labels.
        """
        # Convert the first 20 rows of the column into a single text block
        # We use newlines to help spaCy see them as separate sentences/entries
        sample_text = "\n".join(series.dropna().astype(str).head(20).tolist())
        
        # Process the text through the AI pipeline
        doc = self.nlp(sample_text)
        
        # Extract labels for specific PII categories
        # PERSON/GPE come from spaCy NER; EMAIL/PHONE/IBAN come from our EntityRuler
        pii_categories = ["PERSON", "EMAIL", "PHONE", "IBAN", "GPE"]
        
        found_labels = set([
            ent.label_ for ent in doc.ents 
            if ent.label_ in pii_categories
        ])
        
        return list(found_labels)

# --- End of File ---