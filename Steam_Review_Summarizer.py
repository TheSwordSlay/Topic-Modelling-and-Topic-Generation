import streamlit as st
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.preprocessing import normalize
import scipy.sparse as sp
from llm import generate_response
from torch import bfloat16
import transformers
from torch import cuda
from datasets import Dataset
from sklearn.cluster import KMeans

st.title("Review Summarizer")

uploaded_csv = st.file_uploader(label="Upload reviews CSV here", type=["csv"])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    if 'ReviewText' in df.columns:
        data = df[['ReviewText']]
        data.dropna(inplace=True)
        class NormalizedClassTfidfTransformer(ClassTfidfTransformer):
            def transform(self, X):
                # Perform regular c-TF-IDF transformation
                X_transformed = super().transform(X)
                
                # Apply L2 normalization
                X_normalized = normalize(X_transformed, norm='l2', axis=1)
                
                return sp.csr_matrix(X_normalized)
        ctfidf_model = NormalizedClassTfidfTransformer()
        vectorizer_model = CountVectorizer(stop_words="english")
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = KMeans(n_clusters=18, random_state=42)
        topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model, verbose=True)
        data_array = data.to_numpy()
        data_string = []
        for x in data_array:
            data_string.append(x[0].replace("Product received for free", "").replace("Product refunded", ""))

        dataset_object = Dataset.from_dict({"text": data_string})

        @st.cache_data
        def get_topic_model_dict() :
            topics, probs = topic_model.fit_transform(dataset_object["text"])
            topic_represent = topic_model.get_topic_info()[['Topic', 'Representation']]
            dictionary = topic_represent.set_index('Topic')['Representation'].to_dict()
            return dictionary, topic_model
        
        dictionary, topic_model = get_topic_model_dict()

        def create_topic_prompt(topic_model: BERTopic, topic_id: int, data_string: list) -> str:
            # Get document info with the original documents
            doc_info = topic_model.get_document_info(data_string)
            
            # Filter for the specific topic and get top documents by probability
            topic_docs = doc_info[doc_info['Topic'] == topic_id]
            selected_docs = topic_docs['Document'].tolist()
            
            # Format documents as a list
            formatted_docs = "\n".join(f"- {doc}" for doc in selected_docs)
            
            # Get keywords (top words) for the topic
            keywords = [word for word, _ in topic_model.get_topic(topic_id)]
            formatted_keywords = ", ".join(keywords)
            
            # Create the prompt
            prompt = f"""I have a topic that contains the following documents:
        {formatted_docs}

        The topic is described by the following keywords: '{formatted_keywords}'.

        Based on the information about the topic above, please create a short label of this topic. Important : Make sure you to only return the label and nothing more."""
            
            return prompt

        @st.cache_data
        def generate_topic(topic_id, data_review):  # Replace with your desired topic ID
            prompt = create_topic_prompt(topic_model, topic_id, data_review)

            system_prompt_qwen2 = """<|im_start|>system
            You are a helpful, respectful and honest assistant for labeling topics.<|im_end|>"""

            example_prompt_qwen2 = """<|im_start|>user
            """+prompt+"""<|im_end|>"""

            prompt_qwen2 = system_prompt_qwen2 + example_prompt_qwen2
            res = generate_response(prompt_qwen2)
            return res

        def get_topic_documents(topic_model: BERTopic, topic_id: int, data_string: list) -> list:
            doc_info = topic_model.get_document_info(data_string)
            
            topic_docs = doc_info[doc_info['Topic'] == topic_id]['Document'].tolist()
            
            return topic_docs

        topic_info = topic_model.get_topic_info()

        num_topics = topic_info.shape[0]
        array_topic_generation = ["aaa" for _ in range(num_topics + 5)]

        max_items = 20
        for i, (key, value) in enumerate(dictionary.items()):
            if i >= max_items:
                break

            array_topic = get_topic_documents(topic_model, key, data_string)
            st.write("Topic : "+generate_topic(key, data_string))
            with st.expander(f"See reviews"):
                st.write(f"Keyphrase : {', '.join(value)}")
                st.write("Reviews")

                st.divider() 
                for i, item in enumerate(array_topic):
                    if i >= max_items:
                        break
                    st.write(f"{item}")
                    st.divider() 

    else: 
        st.error("Error : CSV does not contain ReviewText column")
    