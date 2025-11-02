# AI Agent Learning Projects

A comprehensive collection of AI agent implementations and applications using cutting-edge Large Language Models (LLMs), vector databases, and agentic frameworks. This repository contains multiple Jupyter notebooks exploring different AI agent architectures and use cases.

## 📚 Project Overview

This repository contains hands-on implementations of various AI agents and applications, ranging from simple chatbots to sophisticated research agents with RAG (Retrieval-Augmented Generation) capabilities.

## 🚀 Projects Included

### 1. **LangGraph Chatbot** (`langgraph_chatbot.ipynb`)

A conversational AI chatbot built with LangGraph featuring:

- Basic chatbot implementation with state management
- Integration with Tavily AI for web search capabilities
- Tool-based architecture allowing the chatbot to search for real-time information
- Memory persistence using SQLite and MemorySaver for conversation history
- Thread-based conversation management
- Graph visualization with Mermaid diagrams

**Key Features:**

- State-based conversation flow
- Web search integration via Tavily
- Persistent memory across sessions
- Visual graph representation

### 2. **Essay Writer Agent** (`essay_writer.ipynb`)

An intelligent essay writing system using LangGraph that:

- Plans essay structure with high-level outlines
- Conducts research using Tavily web search API
- Generates well-structured 5-paragraph essays
- Implements reflection and critique pattern
- Performs iterative revisions based on feedback
- Supports configurable revision cycles

**Workflow:**

1. Planning → Research → Generation
2. Reflection → Critique → Research → Revision
3. Multiple iterations until quality threshold met

### 3. **Question-Answering on Private Documents** (`main.ipynb`)

A comprehensive RAG (Retrieval-Augmented Generation) system that:

- Loads documents from multiple sources (PDF, DOCX, Wikipedia)
- Chunks and processes documents efficiently
- Creates embeddings using OpenAI's text-embedding models
- Supports two vector database backends:
  - **Pinecone** (cloud-based)
  - **Chroma** (local)
- Implements conversational retrieval with memory
- Supports custom prompts and multi-language responses

**Capabilities:**

- Document loading and chunking
- Vector store creation and management
- Similarity search with metadata filtering
- Chat history for contextual conversations
- Cost estimation for embeddings

### 4. **Pinecone Vector Database** (`pinecone.ipynb`)

Deep dive into Pinecone vector database operations:

- Index creation and management
- Vector upsert, fetch, query, and delete operations
- Namespace management for vector organization
- Serverless deployment configuration
- Advanced query operations with metadata filtering

### 5. **ReAct Agent** (`react_agent.ipynb`)

A from-scratch implementation of the ReAct (Reasoning + Acting) pattern:

- Custom Agent class with system prompts
- Thought-Action-Observation loop
- Multiple tool integrations:
  - Calculator for mathematical operations
  - Cost lookup tool
  - Wikipedia search
- Demonstrates the fundamental ReAct architecture

**Pattern:**

```
Thought → Action → PAUSE → Observation → Answer
```

### 6. **Reflection Pattern** (`reflection.ipynb`)

A tweet generation system implementing the reflection pattern:

- Generate initial tweets based on topics
- Reflect and critique generated content
- Iteratively improve through feedback loops
- State graph implementation with LangGraph
- Configurable iteration limits
- Demonstrates self-improvement through critique

**Components:**

- Generation chain with custom prompts
- Reflection chain for critique
- Message state management
- Conditional edges for flow control

### 7. **Research Agent** (`research_agent.ipynb`)

The most advanced project - a comprehensive AI research agent featuring:

**Data Collection:**

- ArXiv paper extraction and dataset creation
- PDF downloading and processing
- Document chunking with metadata preservation

**Knowledge Base:**

- Embedding generation using OpenAI models
- Pinecone vector database integration
- Efficient batch processing with progress tracking

**Tools Implemented:**

- `fetch_arxiv`: Retrieves paper abstracts by ArXiv ID
- `web_search`: Google SerpAPI integration for web searches
- `rag_search`: Semantic search across AI research papers
- `rag_search_filter`: Filtered search by specific ArXiv ID
- `final_answer`: Structured research report generation

**Agent Architecture:**

- Oracle LLM for decision-making (GPT-4)
- State graph with conditional routing
- Tool selection and execution pipeline
- Scratchpad for tracking research steps
- Automatic report formatting

**Output:**
Generates comprehensive research reports with:

- Introduction
- Research steps taken
- Main body with detailed findings
- Conclusion
- Source citations

## 🛠️ Technologies Used

### Core AI/ML Frameworks

- **LangChain**: Framework for building LLM applications
- **LangGraph**: State graph orchestration for agentic workflows
- **OpenAI API**: GPT-4, GPT-4o-mini, GPT-5-mini models
- **Pydantic**: Data validation and structured outputs

### Vector Databases

- **Pinecone**: Cloud-based vector database with serverless deployment
- **Chroma**: Local vector database for embeddings

### Search & Retrieval

- **Tavily**: AI-powered web search API
- **SerpAPI**: Google search integration
- **Semantic Router**: Encoding and semantic search capabilities

### Document Processing

- **PyPDF**: PDF document loading and parsing
- **docx2txt**: Word document processing
- **Wikipedia**: Wikipedia article retrieval
- **tiktoken**: Token counting and cost estimation

### Data Handling

- **Pandas**: Data manipulation and analysis
- **Requests**: HTTP requests for API interactions
- **Python-dotenv**: Environment variable management

### Utilities

- **tqdm**: Progress bars for batch operations
- **XML ElementTree**: ArXiv API response parsing
- **IPython**: Interactive visualizations

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Dromediansk/ai_agent_learning.git
cd ai_agent_learning
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
   Create a `.env` file in the project root with the following:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
TAVILY_API_KEY=your_tavily_api_key
SERPAPI_KEY=your_serpapi_key
```

## 🎯 Use Cases

- **Document Q&A**: Query private document collections with natural language
- **Research Automation**: Automated literature review and research synthesis
- **Content Generation**: AI-assisted essay and tweet writing with quality control
- **Conversational AI**: Build chatbots with memory and web search capabilities
- **Knowledge Management**: Create searchable knowledge bases from documents

## 📁 Project Structure

```
ztm_ai_apps_learning/
├── essay_writer.ipynb           # Essay writing agent with reflection
├── langgraph_chatbot.ipynb      # Conversational chatbot with tools
├── main.ipynb                   # RAG system for private documents
├── pinecone.ipynb               # Vector database operations
├── react_agent.ipynb            # ReAct pattern implementation
├── reflection.ipynb             # Reflection pattern for content improvement
├── research_agent.ipynb         # Advanced research agent with RAG
├── requirements.txt             # Python dependencies
└── files/
    └── arxiv_dataset.json       # ArXiv research paper dataset
```

## 🔑 Key Concepts Demonstrated

1. **Agentic Workflows**: State machines and decision-making pipelines
2. **RAG Systems**: Retrieval-Augmented Generation for grounded responses
3. **Tool Integration**: LLMs using external tools and APIs
4. **Memory Management**: Conversation history and context preservation
5. **Reflection Pattern**: Self-improvement through critique and revision
6. **Vector Search**: Semantic similarity search in high-dimensional spaces
7. **Prompt Engineering**: System prompts and chain construction
8. **Graph-based Execution**: LangGraph for complex agent behaviors

## 📊 Data Sources

- **ArXiv**: Academic papers in computer science and AI
- **Wikipedia**: General knowledge articles
- **Custom PDFs**: Private documents and research papers
- **Web Search**: Real-time information via Tavily and SerpAPI

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests
- Share improvements to existing implementations

## 📝 License

This project is part of a learning journey and is available for educational purposes.

## 🙏 Acknowledgments

Built while learning AI agent development through hands-on implementation of cutting-edge patterns and architectures.

## 📞 Contact

For questions or discussions about the project, please open an issue in the repository.

---

**Note**: These projects are for educational purposes and demonstrate various AI agent architectures and patterns. API keys are required for full functionality.
