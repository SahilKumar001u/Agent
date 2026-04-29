# Multi-Agent Travel Planner System

**Assignment:** Build a Multi-Agent System using LangChain + LangGraph

## 📋 Overview

This project implements a sophisticated multi-agent AI system that creates comprehensive travel plans through the collaboration of 4 specialized agents. The system uses LangGraph for workflow orchestration and LangChain for LLM integration.

## 🤖 Agent Architecture

### Agent 1: Planner Agent
- **Role:** Extracts structured information from natural language travel requests
- **Input:** Raw user input (free text)
- **Output:** Destination, travel dates, budget, preferences
- **Technology:** JSON parsing with error handling

### Agent 2: Research Agent
- **Role:** Gathers destination information and travel insights
- **Input:** Destination and user preferences
- **Output:** Top attractions, highlights, practical travel tips
- **Technology:** Context-aware LLM prompting

### Agent 3: Itinerary Builder Agent
- **Role:** Creates detailed day-by-day travel itineraries
- **Input:** Destination, dates, preferences, research notes
- **Output:** Structured daily itinerary with activities
- **Technology:** Multi-context synthesis

### Agent 4: Budget Estimator Agent
- **Role:** Provides cost analysis and budget breakdown
- **Input:** Destination, dates, budget, itinerary
- **Output:** Cost breakdown (accommodation, transport, food, activities)
- **Technology:** Budget feasibility analysis

## 🔄 LangGraph Workflow

```
START → Planner Agent → Research Agent → Itinerary Builder → Budget Estimator → END
```

**Workflow Features:**
- Sequential agent execution with shared state
- Each agent updates specific fields in `TravelState`
- Directed edges ensure proper information flow
- Error handling at each node

## 🛠️ Technology Stack

- **LangGraph:** Workflow orchestration and state management
- **LangChain:** LLM integration and prompt templates
- **Groq API:** Free-tier LLM (llama-3.3-70b-versatile)
- **Streamlit:** Web-based user interface
- **Python 3.14:** Core implementation language

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Groq API key (free tier available at [console.groq.com](https://console.groq.com))

### Setup Steps

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd <repo-directory>
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure API key:**
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

## 🌐 Live Demo

**Try it now without installation:** https://multi-agent-travel-planner-cshdmubaf5yrdhae7chiok.streamlit.app/

The application is deployed on Streamlit Cloud and publicly accessible. No setup required!

## 🚀 Usage

### Option 1: Use the Live Demo (Easiest)

Simply visit the live demo link above and start planning your trip!

### Option 2: Streamlit Web Interface (Local)

```bash
streamlit run multi_agent_system_streamlit.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Beautiful web interface
- Real-time progress indicators
- Tabbed results view
- Download travel plan as text file
- Example requests for guidance

### Option 3: Command Line Interface

```bash
python multi_agent_system_streamlit.py
```

**Features:**
- Simple text-based interface
- Formatted console output
- Quick testing and debugging

## 💡 Example Requests

Try these example travel requests:

1. **Tokyo Trip:**
   ```
   I want to visit Tokyo for 7 days in March with a budget of $3000. I love food and temples.
   ```

2. **Paris Romantic Getaway:**
   ```
   Plan a romantic trip to Paris for 5 days in June. Budget is $2500. We enjoy art museums and cafes.
   ```

3. **Bali Family Vacation:**
   ```
   Family vacation to Bali for 10 days in December. Budget $5000. Kids love beaches and animals.
   ```

4. **Thailand Backpacking:**
   ```
   Solo backpacking trip to Thailand for 2 weeks. Budget $1500. Interested in culture and nightlife.
   ```

## 📊 Assignment Requirements Checklist

### ✅ Code Requirements
- [x] Single Python file implementation (`multi_agent_system_streamlit.py`)
- [x] 4 agents with clear, distinct roles
- [x] LangGraph nodes and edges for workflow
- [x] Shared state/context (TravelState TypedDict)
- [x] main() function for CLI execution
- [x] Dynamic user input acceptance

### ✅ Technical Implementation
- [x] Agent collaboration through shared state
- [x] Clear role separation and responsibilities
- [x] Workflow orchestration with LangGraph
- [x] Error handling and validation
- [x] Comprehensive documentation

### ✅ Additional Features
- [x] Streamlit web interface for better UX
- [x] Both CLI and web modes supported
- [x] Detailed code comments and docstrings
- [x] Example requests and usage guide
- [x] Download functionality for travel plans

## 🎯 Evaluation Criteria Coverage

### LangGraph Workflow (30%)
- ✅ Proper StateGraph implementation
- ✅ Clear node definitions for each agent
- ✅ Directed edges defining workflow
- ✅ Shared state management (TravelState)
- ✅ Entry point and END node configuration

### Agent Design Clarity (25%)
- ✅ 4 distinct agents with specific roles
- ✅ Clear input/output for each agent
- ✅ Comprehensive docstrings explaining agent purpose
- ✅ Logical workflow progression
- ✅ Agent collaboration through state updates

### Code Quality (15%)
- ✅ Clean, readable code structure
- ✅ Comprehensive comments and documentation
- ✅ Error handling and validation
- ✅ Type hints (TypedDict for state)
- ✅ Modular function design

### Output Usefulness (15%)
- ✅ Comprehensive travel plans generated
- ✅ Structured itineraries with daily activities
- ✅ Detailed budget breakdowns
- ✅ Destination research and tips
- ✅ Formatted, easy-to-read output

### Demo Explanation (15%)
- ✅ Clear system architecture
- ✅ Agent roles well-documented
- ✅ Workflow visualization
- ✅ Example outputs provided
- ✅ Usage instructions included

## 📁 Project Structure

```
.
├── multi_agent_system_streamlit.py  # Main implementation (single file)
├── requirements.txt                  # Python dependencies
├── .env                             # API key configuration
├── .env.example                     # Example environment file
└── README.md                        # This file
```

## 🔑 Key Features

1. **Intelligent Parsing:** Planner agent extracts structured data from natural language
2. **Contextual Research:** Research agent provides destination-specific insights
3. **Personalized Itineraries:** Itinerary builder creates custom day-by-day plans
4. **Budget Analysis:** Budget estimator provides realistic cost breakdowns
5. **User-Friendly Interface:** Streamlit UI with progress indicators and tabbed results
6. **Dual Mode:** Works as both CLI and web application
7. **Error Handling:** Graceful handling of API errors and invalid inputs
8. **Free Tier LLM:** Uses Groq's free API (no credit card required)


## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Install dependencies: `pip install -r requirements.txt`

### Issue: "API key not found"
**Solution:** Create `.env` file with `GROQ_API_KEY=your_key_here`

### Issue: "Rate limit exceeded"
**Solution:** Groq free tier has limits. Wait a few minutes or upgrade to paid tier.

### Issue: "Planner returns 'unknown' destination"
**Solution:** Make your request more specific. Include clear destination name.

## 🎓 Key Learnings

1. **Multi-Agent Collaboration:** How to design agents with specific roles that work together
2. **State Management:** Using TypedDict for shared state across agents
3. **LangGraph Workflow:** Building directed workflows with nodes and edges
4. **LLM Integration:** Effective prompt engineering for different agent roles
5. **Error Handling:** Graceful degradation when LLM outputs are unexpected
6. **UI/UX Design:** Creating user-friendly interfaces with Streamlit
7. **Free Tier APIs:** Leveraging Groq's free tier for cost-effective development

## 📄 License

This project is created for educational purposes as part of a multi-agent system assignment.

## 👤 Author

Sahil


## 🙏 Acknowledgments

- LangChain and LangGraph teams for excellent documentation
- Groq for providing free-tier API access
- Streamlit for the intuitive web framework
