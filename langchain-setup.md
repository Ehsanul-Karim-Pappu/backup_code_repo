# LangChain-based Local LLM Setup for EDA Documentation

## 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n local_llm python=3.10
conda activate local_llm

# Install required packages
pip install langchain
pip install langchain-community
pip install chromadb
pip install pypdf
pip install python-dotenv
pip install sentence-transformers
pip install ctransformers
```

## 2. Download the Model
We'll use a CPU-friendly model:

```bash
mkdir models
cd models
wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q3_K_S.gguf
```

## 3. Create the Document Processing Script

Save as `process_eda_docs.py`:

```python
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class EDADocumentProcessor:
    def __init__(self, model_path, docs_path):
        self.model_path = model_path
        self.docs_path = docs_path
        self.db_path = "./db"
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize LLM
        self.llm = CTransformers(
            model=model_path,
            model_type="phi",
            config={
                'max_new_tokens': 512,
                'temperature': 0.1,
                'context_length': 2048,
                'threads': os.cpu_count(),
            }
        )

    def load_and_split_documents(self):
        """Load PDFs and split into chunks"""
        print("Loading documents...")
        
        # Load PDFs from directory
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )
        
        texts = text_splitter.split_documents(documents)
        print(f"Split documents into {len(texts)} chunks")
        return texts

    def create_vector_store(self, texts):
        """Create and persist vector store"""
        print("Creating vector store...")
        
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        vectorstore.persist()
        return vectorstore

    def setup_qa_chain(self, vectorstore):
        """Set up the QA chain with custom prompt"""
        template = """
        You are an expert in EDA (Electronic Design Automation) tools and their scripting languages.
        Use the provided documentation to answer the question. 
        If you don't find an exact answer, say so and provide the closest relevant information.
        Include specific references to documentation sections or pages when possible.
        
        Context: {context}
        
        Question: {question}
        
        Answer: Let me help you with that based on the documentation.
        """
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True
        )
        
        return qa_chain

    def process_documents(self):
        """Main processing function"""
        texts = self.load_and_split_documents()
        vectorstore = self.create_vector_store(texts)
        qa_chain = self.setup_qa_chain(vectorstore)
        return qa_chain

def main():
    # Initialize processor
    processor = EDADocumentProcessor(
        model_path="./models/phi-2.Q3_K_S.gguf",
        docs_path="./documentation"
    )
    
    # Process documents
    processor.process_documents()
    print("Document processing complete! You can now use query_eda_docs.py to ask questions.")

if __name__ == "__main__":
    main()
```

## 4. Create the Query Interface

Save as `query_eda_docs.py`:

```python
from process_eda_docs import EDADocumentProcessor
from langchain_community.vectorstores import Chroma
import gc
import os

class EDAQuerySystem:
    def __init__(self):
        self.model_path = "./models/phi-2.Q3_K_S.gguf"
        self.db_path = "./db"
        
        # Initialize processor
        self.processor = EDADocumentProcessor(
            model_path=self.model_path,
            docs_path="./documentation"
        )
        
        # Load existing vectorstore
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.processor.embeddings
        )
        
        # Setup QA chain
        self.qa_chain = self.processor.setup_qa_chain(self.vectorstore)

    def query(self, question):
        """Process a single query"""
        try:
            result = self.qa_chain({"query": question})
            
            # Format response
            response = result['result']
            sources = result['source_documents']
            
            # Print response
            print("\nResponse:", response)
            print("\nSources:")
            seen_sources = set()
            for doc in sources:
                source = doc.metadata.get('source', '')
                if source and source not in seen_sources:
                    seen_sources.add(source)
                    print(f"- {os.path.basename(source)}")
                    
        except Exception as e:
            print(f"An error occurred: {e}")
            self.cleanup_memory()

    def cleanup_memory(self):
        """Clean up memory"""
        gc.collect()
        print("Memory cleaned up")

    def run_interactive(self):
        """Run interactive query session"""
        print("EDA Documentation Assistant Ready!")
        print("Enter your questions (type 'exit' to quit, 'clear' to free memory)")
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() == 'exit':
                break
            elif question.lower() == 'clear':
                self.cleanup_memory()
                continue
            
            self.query(question)

def main():
    query_system = EDAQuerySystem()
    query_system.run_interactive()

if __name__ == "__main__":
    main()
```

## 5. Directory Structure

```
eda_assistant/
├── models/
│   └── phi-2.Q3_K_S.gguf
├── documentation/
│   ├── synopsys_docs/
│   │   └── *.pdf
│   └── cadence_docs/
│       └── *.pdf
├── db/              # Vector store will be saved here
├── process_eda_docs.py
└── query_eda_docs.py
```

## 6. Usage Instructions

1. First-time setup:
```bash
# Create directory structure
mkdir -p documentation/synopsys_docs documentation/cadence_docs db models

# Copy your documentation
cp path/to/your/docs/synopsys/* documentation/synopsys_docs/
cp path/to/your/docs/cadence/* documentation/cadence_docs/

# Process documents (only needed once or when adding new docs)
python process_eda_docs.py
```

2. Regular usage:
```bash
# Start the query interface
python query_eda_docs.py
```

## 7. Example Queries

```
Your question: How do I measure parasitic capacitance in Custom Compiler?
Your question: What is the SKILL syntax for creating a new cellview in Virtuoso?
Your question: How can I write a TCL script to generate a layout?
```

## Key Features of this LangChain Implementation

1. **Better Document Processing**:
   - Recursive text splitting for better context preservation
   - Handles PDF metadata properly
   - Better handling of technical documentation structure

2. **Improved Memory Management**:
   - More efficient document chunking
   - Better memory cleanup
   - Persistent vector store

3. **Enhanced Query Capabilities**:
   - More accurate source references
   - Better context handling
   - Custom prompting for EDA-specific responses

4. **Easy Maintenance**:
   - Simple to add new documents
   - Clear separation of processing and querying
   - Easy to modify prompt templates

## Performance Tips

1. Document Processing:
   - Group related documents in subdirectories
   - Use meaningful file names
   - Keep PDFs under 100MB each for better processing

2. Query Performance:
   - Be specific in your questions
   - Include relevant context (e.g., tool name, language)
   - Use technical terms from the documentation

3. Memory Management:
   - Use 'clear' command if responses become slow
   - Restart the system after several hours of use
   - Process large document sets in batches

## Troubleshooting

1. Slow Responses:
   - Reduce chunk_size in text_splitter
   - Lower the number of returned sources (k value)
   - Use more specific queries

2. Memory Issues:
   - Use the 'clear' command
   - Restart the application
   - Process fewer documents at once

3. Quality Issues:
   - Adjust the prompt template
   - Increase context length
   - Use more specific queries
