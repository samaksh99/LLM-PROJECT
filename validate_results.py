from sentence_transformers import SentenceTransformer, util
class ValidateResultsInteractor:
    def validate_results(self, search_result: str, input_text: str) -> float:
        """
                Calculate cosine similarity between two text strings using a pre-trained Sentence Transformer model.

                Parameters:
                - search_result (str): The first text string for comparison.
                - input_text (str): The second text string for comparison.

                Returns:
                - float: The cosine similarity score between the two input text strings.
                """
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode([search_result, input_text],
                                  convert_to_tensor=True)
        cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        similarity_score = cosine_sim.item()

        return similarity_score