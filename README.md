# Agent Trip Planner

A professional, agentic AI-powered travel planning application.

## 📁 Project Structure

```text
├── client/                 # Frontend application (blank)
└── server/                 # Backend application
    ├── config/             # Configuration files
    ├── connectors/         # External service connectors (DB, VectorDB, LLM)
    ├── middlewares/        # Express middlewares (Auth, Error, Logger, etc.)
    ├── routes/             # API route handlers
    ├── utils/              # Shared utility functions
    ├── src/                # Core application source
    │   ├── agents/         # AI Agents (Planner, Research, Orchestrator)
    │   ├── database/       # DB Models and schemas
    │   ├── multiRAG/       # Multi-source RAG pipeline
    │   ├── vectordb/       # Vector store logic
    │   ├── webscraping/    # Web crawling and parsing
    │   ├── test/           # Test suites
    │   └── output/         # Generated files and reports
    ├── server.js           # Entry point
    └── package.json        # Dependencies and scripts
```

## 🚀 Getting Started

1. `cd server`
2. `npm install`
3. Fill in `.env` based on `.env.example`
4. `npm run dev`
# agentic-trip-planner
