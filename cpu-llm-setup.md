# CPU-Only Local LLM Setup for EDA Documentation

## 1. System Requirements
- Minimum 16GB RAM (8GB might work but not recommended)
- At least 8GB free storage
- Windows/Linux (Linux preferred for better performance)
- Multi-core CPU (recommended: 4+ cores)

## 2. Environment Setup

```bash
# Create a new conda environment
conda create -n local_llm python=3.10
conda activate local_llm

# Install required packages
pip install llama-cpp-python
pip install sentence-transformers
pip install pypdf
pip install chromadb
pip install llama-index
```

## 3. Download the Model
We'll use a smaller, CPU-optimized model. Here are two options:

Option 1 (Recommended for better quality):
```bash
# Create models directory
mkdir models
cd models

# Download Mistral 7B Instruct 3-bit quantized version
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q3_K_S.gguf
```

Option 2 (Faster, but lower quality):
```bash
# Download Phi-2 model (smaller and faster)
wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q3_K_S.gguf
```

## 4. Create the CPU-Optimized Document Processing Script

Save as `process_docs.py`:

```python
import os
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    set_global_service_context
)
from llama_index.llms import LlamaCPP
from llama_index.embeddings import HuggingFaceEmbeddings
import logging
import sys

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def initialize_llm(model_path):
    # Initialize the LLM with CPU-optimized settings
    llm = LlamaCPP(
        model_path=model_path,
        temperature=0.1,
        max_new_tokens=256,  # Reduced for CPU
        context_window=2048,  # Reduced for CPU
        generate_kwargs={},
        model_kwargs={"n_threads": os.cpu_count()},  # Use all CPU cores
        verbose=True,
    )
    
    # Initialize embeddings (smaller model for CPU)
    embed_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # Smaller embedding model
        cache_folder="./embeddings_cache"
    )
    
    # Create service context with CPU-optimized settings
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        chunk_size=256,  # Smaller chunks for CPU
        chunk_overlap=20
    )
    
    return service_context

def process_documents(docs_dir, service_context):
    """Process documents in batches to manage memory"""
    print(f"Processing documents from {docs_dir}")
    
    # Load and process documents in smaller batches
    for root, _, files in os.walk(docs_dir):
        pdf_files = [f for f in files if f.endswith('.pdf')]
        
        # Process 2 PDFs at a time
        for i in range(0, len(pdf_files), 2):
            batch_files = pdf_files[i:i+2]
            print(f"Processing batch: {batch_files}")
            
            documents = SimpleDirectoryReader(
                input_files=[os.path.join(root, f) for f in batch_files],
                filename_as_id=True
            ).load_data()
            
            # Create or update index
            if i == 0:
                index = VectorStoreIndex.from_documents(
                    documents,
                    service_context=service_context
                )
            else:
                index.insert_nodes(documents)
            
            # Save after each batch
            index.storage_context.persist("./storage")
            print(f"Batch {i//2 + 1} completed and saved")
    
    return index

def create_query_engine(index):
    """Create a memory-efficient query engine"""
    return index.as_query_engine(
        response_mode="compact",  # More memory efficient
        verbose=True,
        similarity_top_k=2  # Reduced for CPU
    )

if __name__ == "__main__":
    # Initialize
    model_path = "./models/mistral-7b-instruct-v0.2.Q3_K_S.gguf"  # or phi-2.Q3_K_S.gguf
    service_context = initialize_llm(model_path)
    set_global_service_context(service_context)
    
    # Process documents
    docs_dir = "./documentation"
    index = process_documents(docs_dir, service_context)
```

## 5. Create the Memory-Efficient Query Interface

Save as `query_docs.py`:

```python
from llama_index import StorageContext, load_index_from_storage
from process_docs import initialize_llm, create_query_engine
import gc
import os

def load_existing_index(model_path):
    # Initialize LLM and service context
    service_context = initialize_llm(model_path)
    
    # Load existing index
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    
    return index

def main():
    model_path = "./models/mistral-7b-instruct-v0.2.Q3_K_S.gguf"  # or phi-2.Q3_K_S.gguf
    
    print("Loading index... This might take a minute...")
    index = load_existing_index(model_path)
    query_engine = create_query_engine(index)
    
    print("\nEDA Documentation Assistant Ready!")
    print("Enter your questions (type 'exit' to quit)")
    print("Type 'clear' to free up memory if responses become slow")
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() == 'exit':
            break
        
        if question.lower() == 'clear':
            gc.collect()  # Force garbage collection
            print("Memory cleared!")
            continue
            
        try:
            response = query_engine.query(question)
            print("\nResponse:", response.response)
            print("\nSources:")
            for source_node in response.source_nodes:
                print(f"- {source_node.node.metadata['file_name']}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Try clearing memory by typing 'clear'")

if __name__ == "__main__":
    main()
```

## 6. Usage Instructions

1. Initial Setup:
```bash
# Set up directory structure
mkdir -p documentation/synopsys_docs documentation/cadence_docs storage models embeddings_cache

# Copy your documentation
cp path/to/your/docs/synopsys/* documentation/synopsys_docs/
cp path/to/your/docs/cadence/* documentation/cadence_docs/
```

2. First-time processing:
```bash
# Process documents (this might take a while on CPU)
python process_docs.py
```

3. Regular usage:
```bash
# Start the query interface
python query_docs.py
```

## Performance Tips for CPU-Only Systems

1. Memory Management:
   - Close other applications while using the system
   - Use the 'clear' command in the query interface if responses become slow
   - Process large documentation sets overnight

2. Improving Response Times:
   - Keep individual PDF files under 100 pages if possible
   - Split large manuals into smaller documents
   - Use descriptive filenames for better reference

3. Optimizing Resource Usage:
   - Adjust `chunk_size` based on your RAM (lower if you have less RAM)
   - Reduce `max_new_tokens` for faster responses
   - Use the Phi-2 model if Mistral is too slow

## Troubleshooting

1. If you get "Out of Memory" errors:
   - Reduce `chunk_size` in `process_docs.py`
   - Use the smaller Phi-2 model
   - Process fewer documents at once

2. If responses are too slow:
   - Try the Phi-2 model instead of Mistral
   - Reduce `context_window` size
   - Use shorter queries

3. If responses lack detail:
   - Increase `max_new_tokens`
   - Try more specific questions
   - Consider splitting complex queries into smaller parts
