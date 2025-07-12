# Grocery AI Assistant using Retrieval-Augmented Generation (RAG)

This repository contains code and resources for building a fully local AI assistant that answers natural language questions about grocery products using the Instacart Market Basket Analysis dataset. The assistant uses semantic search with FAISS and a local language model (Flan-T5) to provide relevant answers without relying on cloud APIs.

---

## Project Overview

* **Dataset:** Instacart Market Basket Analysis (Kaggle)
* **Embedding model:** sentence-transformers/all-MiniLM-L6-v2
* **Vector store:** FAISS
* **Local LLM:** google/flan-t5-base
* **Interface:** Gradio

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline with local models, enabling semantic search and question answering on grocery data.

---

## Installation

```bash
pip install -r requirements.txt
```

Required packages include: langchain, langchain-community, faiss-cpu, sentence-transformers, transformers, accelerate, gradio, kagglehub.

---

## Usage

### 1. Download and Load Dataset

```python
import os
import pandas as pd
import kagglehub

print("Starting dataset download...")
data_path = kagglehub.dataset_download("yasserh/instacart-online-grocery-basket-analysis-dataset")
print(f"Dataset downloaded and extracted at: {data_path}")

products_df = pd.read_csv(os.path.join(data_path, "products.csv"))
aisles_df = pd.read_csv(os.path.join(data_path, "aisles.csv"))
departments_df = pd.read_csv(os.path.join(data_path, "departments.csv"))
```

### 2. Merge and Prepare Product Metadata

```python
products_full = products_df.merge(aisles_df, how="left", on="aisle_id")
products_full = products_full.merge(departments_df, how="left", on="department_id")

products_full["combined_text"] = products_full.apply(
    lambda row: f"Product: {row['product_name']}. Aisle: {row['aisle']}. Department: {row['department']}.", axis=1
)
```

### 3. Create Embeddings and Vector Store

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

local_embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(products_full["combined_text"].tolist(), local_embedder)
```

### 4. Load Local LLM and Define QA Function

```python
from transformers import pipeline

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)

def local_answer_question(query):
    docs = vector_store.similarity_search(query, k=4)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
    result = qa_pipeline(prompt)[0]["generated_text"]
    return result
```

### 5. Run Gradio Interface

```python
import gradio as gr

chat_interface = gr.Interface(
    fn=local_answer_question,
    inputs="text",
    outputs="text",
    title="Grocery AI Assistant (Fully Local)",
    description="Ask about grocery products — answers generated using FAISS + Flan-T5."
)

chat_interface.launch(share=True, debug=True)
```

---

## Project Structure

* `products.csv`, `aisles.csv`, `departments.csv`: Dataset files
* `embedding.py`: Code to create embeddings and vector store
* `qa.py`: Local question-answering logic
* `app.py`: Gradio web interface

---

## Skills Demonstrated

* Data loading and cleaning with Pandas
* Semantic embeddings and vector similarity search
* Local deployment of open-source language models
* Building an interactive web app with Gradio
* Retrieval-Augmented Generation (RAG) pipeline implementation

---

## Live Demo

Try the live demo hosted on Hugging Face Spaces:

[https://huggingface.co/spaces/kmsmohamedansar/ai\_knowledge\_assistant](https://huggingface.co/spaces/kmsmohamedansar/ai_knowledge_assistant)



