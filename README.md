AI-KnowMap Prototype

Streamlit-powered Knowledge Mapping System that converts unstructured text into a structured Knowledge Graph with semantic search capabilities.

🚀 Overview

AI-KnowMap Prototype enables end-to-end knowledge extraction from raw text by integrating:

Module	Function
Dataset Loader	Upload & preprocess textual data
NLP Pipeline	Named Entity Recognition + Triple Extraction
Graph Builder	Generate & interact with Knowledge Graph
Semantic Search	Query & visualize related entities
Admin Dashboard	Show statistics + collect feedback

The entire system runs in a Streamlit Web Application 🚀

🧱 Core Workflow

1️⃣ Upload raw dataset (CSV)
2️⃣ NLP extracts Subject-Relation-Object triples
3️⃣ Generate interactive Knowledge Graph
4️⃣ Perform semantic search to explore relationships
5️⃣ Monitor analytics via Dashboard

📁 Recommended Project Structure
AI-KnowMap/
│
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── modules/
│   ├── nlp_pipeline.py
│   ├── graph_builder.py
│   └── semantic_search.py
├── requirements.txt
└── README.md

⚙️ Installation & Setup
1️⃣ Create Virtual Environment
python -m venv proj_env


Activate environment:

# Linux/macOS
source proj_env/bin/activate

# Windows
.\proj_env\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

▶️ Run the Streamlit App
streamlit run app.py


Open in browser:
👉 http://localhost:8501/

📌 Data Format Requirement

Input CSV must contain a text column.

Example Input:

ID	Text
1	Alice works at ACME Corp.

Extracted Triples:

Subject	Relation	Object
Alice	works_at	ACME Corp.
🌐 Knowledge Graph Features

✔️ Directed graph visualization
✔️ Subjects = Blue nodes
✔️ Objects = Green nodes
✔️ Hover tooltips + zoom + drag
✔️ Exportable as HTML

Output saved as:
cross_domain_knowledge_graph.html

🔍 Semantic Search

Query entities/keywords

Display related triples

Produce filtered mini-graphs

📊 Admin Dashboard

Entity statistics

Relation count visualization

Feedback submission (demo)

🛠 Technology Stack
Technology	Purpose
Streamlit	Web UI
spaCy	NLP & Entity Extraction
NetworkX + PyVis	Knowledge Graph Visualization
Pandas	Data Handling
👥 Team Contributions
Person	Role
Person 1	Git & Repo Management
Person 2	Data Collection & Preprocessing
Person 3	NLP Pipeline Development
Person 4	Graph Builder & Visualization
Person 5	Semantic Search Integration
📜 License

MIT License © 2025 AI-KnowMap Team
