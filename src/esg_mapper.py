from sentence_transformers import SentenceTransformer, util
import torch

class ESGMapper:
    def __init__(self):
        # Force CPU to avoid the PyTorch 2.4+ requirement error
        device = "cpu"
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        self.esrs_taxonomy = {
            "ESRS E1": "Climate change and greenhouse gas emissions",
            "ESRS E2": "Pollution of air, water, and soil",
            "ESRS E3": "Water and marine resources",
            "ESRS S1": "Own workforce and social equality"
        }
        self.taxonomy_labels = list(self.esrs_taxonomy.keys())
        self.taxonomy_descriptions = list(self.esrs_taxonomy.values())
        self.taxonomy_embeddings = self.model.encode(self.taxonomy_descriptions, convert_to_tensor=True)

    def map_columns(self, df_columns):
        mappings = {}
        for col in df_columns:
            col_embedding = self.model.encode(col, convert_to_tensor=True)
            cos_scores = util.cos_sim(col_embedding, self.taxonomy_embeddings)[0]
            best_match_idx = torch.argmax(cos_scores).item()
            
            if cos_scores[best_match_idx] > 0.35:
                mappings[col] = {
                    "ESRS_Code": self.taxonomy_labels[best_match_idx],
                    "Confidence": round(float(cos_scores[best_match_idx]), 2)
                }
        return mappings