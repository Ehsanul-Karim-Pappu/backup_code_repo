# Setting Up Local LLM for EDA Documentation

## 1. Environment Setup

```bash
# Create a new conda environment
conda create -n local_llm python=3.10
conda activate local_llm

# Install required packages
pip install llama-index
pip install PyPDF2
pip install transformers
pip install torch torchvision torchaudio
pip install sentence-transformers
pip install chromadb
```

## 2. Download the LLM

We'll use Mistral 7B Instruct, which offers a good balance of performance and resource usage:

```bash
# Create a models directory
mkdir models
cd models

# Download Mistral 7B Instruct
git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

# The 4-bit quantized version is recommended for desktop use
# Copy the q4_K_M.gguf file to your working directory
```

## 3. Create the Document Processing Script

Save this as `process_docs.py`:

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

def initialize_llm():
    # Initialize the LLM
    llm = LlamaCPP(
        model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        temperature=0.1,
        max_new_tokens=512,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": -1},
        verbose=True,
    )
    
    # Initialize embeddings
    embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
    )
    
    # Create service context
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        chunk_size=512,
    )
    
    return service_context

def process_documents(docs_dir):
    # Load documents
    documents = SimpleDirectoryReader(
        docs_dir,
        filename_as_id=True,
        recursive=True,
        required_exts=['.pdf']
    ).load_data()
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )
    
    # Save index
    index.storage_context.persist("./storage")
    
    return index

def create_query_engine(index):
    return index.as_query_engine(
        response_mode="tree_summarize",
        verbose=True,
        similarity_top_k=3
    )

if __name__ == "__main__":
    # Initialize
    service_context = initialize_llm()
    set_global_service_context(service_context)
    
    # Process documents
    docs_dir = "./documentation"  # Your PDF directory
    index = process_documents(docs_dir)
    
    # Create query engine
    query_engine = create_query_engine(index)
```

## 4. Create the Query Interface

Save this as `query_docs.py`:

```python
from llama_index import StorageContext, load_index_from_storage
from process_docs import initialize_llm, create_query_engine

def load_existing_index():
    # Initialize LLM and service context
    service_context = initialize_llm()
    
    # Load existing index
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    
    return index

def main():
    # Load index
    index = load_existing_index()
    query_engine = create_query_engine(index)
    
    print("EDA Documentation Assistant Ready!")
    print("Enter your questions (type 'exit' to quit)")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break
            
        response = query_engine.query(question)
        print("\nResponse:", response.response)
        print("\nSources:")
        for source_node in response.source_nodes:
            print(f"- {source_node.node.metadata['file_name']}, "
                  f"Score: {source_node.score:.2f}")

if __name__ == "__main__":
    main()
```

## 5. Directory Structure

```
your_project/
├── models/
│   └── mistral-7b-instruct-v0.2.Q4_K_M.gguf
├── documentation/
│   ├── synopsys_docs/
│   │   └── *.pdf
│   └── cadence_docs/
│       └── *.pdf
├── storage/
├── process_docs.py
└── query_docs.py
```

## 6. Usage Instructions

1. First-time setup:
```bash
# Create directory structure
mkdir -p documentation/synopsys_docs documentation/cadence_docs storage

# Copy your PDF documentation
cp path/to/your/docs/synopsys/* documentation/synopsys_docs/
cp path/to/your/docs/cadence/* documentation/cadence_docs/

# Process documents (only needed once or when adding new docs)
python process_docs.py
```

2. Regular usage:
```bash
# Start the query interface
python query_docs.py
```

## 7. Example Queries

```
Your question: How do I create a new layout cell in Virtuoso using SKILL?

Your question: What's the syntax for measuring parasitic capacitance in Custom Compiler?
```

## Performance Tips

1. GPU Acceleration:
   - Enable GPU acceleration by setting `n_gpu_layers=-1` in LlamaCPP initialization
   - For better performance, use CUDA-enabled PyTorch

2. Memory Usage:
   - Adjust `chunk_size` in service_context based on your RAM
   - Use quantized models (Q4_K_M) for better memory efficiency

3. Response Quality:
   - Adjust `temperature` for more/less focused responses
   - Modify `similarity_top_k` to consider more/fewer source documents

## Troubleshooting

1. If you get CUDA/GPU errors:
   - Ensure NVIDIA drivers are up to date
   - Try reducing `n_gpu_layers`
   - Fall back to CPU by setting `n_gpu_layers=0`

2. If responses are slow:
   - Reduce `chunk_size`
   - Use a smaller model variant
   - Increase GPU memory allocation

3. If responses lack detail:
   - Increase `max_new_tokens`
   - Increase `similarity_top_k`
   - Reduce `temperature`
