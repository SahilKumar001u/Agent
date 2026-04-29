"""
Multi-Agent Travel Planner System
Assignment: Build a Multi-Agent System using LangChain + LangGraph

This system uses 4 specialized agents to create comprehensive travel plans:
1. Planner Agent - Extracts structured information from user requests
2. Research Agent - Gathers destination information and attractions
3. Itinerary Builder Agent - Creates day-by-day travel itineraries
4. Budget Estimator Agent - Provides cost breakdowns and budget analysis

The agents collaborate through a shared state (TravelState) and are orchestrated
using LangGraph's directed workflow.

Usage:
    CLI Mode: python multi_agent_system_streamlit.py
    Streamlit Mode: streamlit run multi_agent_system_streamlit.py
"""

import os
import sys
import json
from typing import TypedDict
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# SHARED STATE DEFINITION
# ============================================================================

class TravelState(TypedDict):
    """
    Shared state passed between all agents in the workflow.
    Each agent reads from and writes to specific fields.
    """
    user_input: str          # Raw free-text from the user
    destination: str         # Extracted by Planner; "unknown" if not found
    travel_dates: str        # Extracted by Planner; empty string if not found
    budget: str              # Extracted by Planner; empty string if not found
    preferences: str         # Extracted by Planner; empty string if not found
    research_notes: str      # Populated by Research Agent
    itinerary: str           # Populated by Itinerary Builder Agent
    budget_estimate: str     # Populated by Budget Estimator Agent


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Get API key from environment or Streamlit secrets
def get_api_key():
    """Get Groq API key from environment variables or Streamlit secrets"""
    # Try to get from environment first (local development)
    api_key = os.getenv("GROQ_API_KEY")
    
    # If not found, try Streamlit secrets (cloud deployment)
    if not api_key:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
        except:
            pass
    
    return api_key

# Shared LLM instance — API key is read from GROQ_API_KEY env var or Streamlit secrets
# Using Groq's free tier with llama-3.3-70b-versatile model
api_key = get_api_key()
if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please set it in:\n"
        "- Local: .env file with GROQ_API_KEY=your_key\n"
        "- Streamlit Cloud: App Settings → Secrets → Add GROQ_API_KEY = \"your_key\""
    )

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, api_key=api_key)


# ============================================================================
# AGENT 1: PLANNER AGENT
# ============================================================================

def planner_node(state: TravelState) -> dict:
    """
    Planner Agent: Extracts structured travel information from user input.
    
    Role: Parse free-text travel requests and extract:
        - Destination
        - Travel dates
        - Budget
        - Preferences (activities, interests, etc.)
    
    Input: user_input from TravelState
    Output: Updates destination, travel_dates, budget, preferences in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a travel planning assistant. Extract structured travel intent from the user's request.\n"
         "Return ONLY a valid JSON object with these exact keys: destination, travel_dates, budget, preferences.\n"
         "Do not include any other text, explanations, or markdown formatting.\n"
         "If you cannot determine a value, use an empty string. If destination is unclear, use \"unknown\".\n\n"
         "Example output:\n"
         '{{"destination": "Paris", "travel_dates": "June 15-22", "budget": "$2000", "preferences": "art museums and cafes"}}'),
        ("human", "Travel request: {user_input}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"user_input": state["user_input"]})
    
    try:
        # Clean up response - remove markdown code blocks if present
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        parsed = json.loads(content)
        return {
            "destination": parsed.get("destination", "unknown"),
            "travel_dates": parsed.get("travel_dates", ""),
            "budget": parsed.get("budget", ""),
            "preferences": parsed.get("preferences", ""),
        }
    except Exception as e:
        print(f"Warning: Failed to parse planner response: {e}")
        print(f"Raw response: {response.content}")
        return {"destination": "unknown", "travel_dates": "", "budget": "", "preferences": ""}


# ============================================================================
# AGENT 2: RESEARCH AGENT
# ============================================================================

def research_node(state: TravelState) -> dict:
    """
    Research Agent: Gathers destination information and travel tips.
    
    Role: Research the destination and provide:
        - Top attractions and highlights
        - Practical travel tips
        - Local insights based on user preferences
    
    Input: destination, preferences from TravelState
    Output: Updates research_notes in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a destination research specialist. Provide at least 3 highlights, top attractions,\n"
         "and practical travel tips for the given destination and traveller preferences.\n"
         "If destination is \"unknown\", provide general travel tips instead."),
        ("human", "Destination: {destination}\nPreferences: {preferences}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "preferences": state["preferences"],
    })
    return {"research_notes": response.content}


# ============================================================================
# AGENT 3: ITINERARY BUILDER AGENT
# ============================================================================

def itinerary_node(state: TravelState) -> dict:
    """
    Itinerary Builder Agent: Creates detailed day-by-day travel plans.
    
    Role: Build a structured itinerary that:
        - Organizes activities by day
        - Incorporates research findings
        - Aligns with user preferences and travel dates
    
    Input: destination, travel_dates, preferences, research_notes from TravelState
    Output: Updates itinerary in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert travel itinerary builder. Create a detailed day-by-day itinerary.\n"
         "Each day must include at least one activity drawn directly from the research notes provided.\n"
         "Format clearly with Day 1, Day 2, etc."),
        ("human",
         "Destination: {destination}\n"
         "Dates: {travel_dates}\n"
         "Preferences: {preferences}\n"
         "Research Notes: {research_notes}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "travel_dates": state["travel_dates"],
        "preferences": state["preferences"],
        "research_notes": state["research_notes"],
    })
    return {"itinerary": response.content}


# ============================================================================
# AGENT 4: BUDGET ESTIMATOR AGENT
# ============================================================================

def budget_node(state: TravelState) -> dict:
    """
    Budget Estimator Agent: Provides cost analysis and budget breakdown.
    
    Role: Analyze the itinerary and provide:
        - Cost breakdown (accommodation, transport, food, activities)
        - Budget feasibility assessment
        - Recommendations for staying within budget
    
    Input: destination, travel_dates, budget, itinerary from TravelState
    Output: Updates budget_estimate in TravelState
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a travel budget analyst. Provide a cost breakdown covering accommodation,\n"
         "transport, food, and activities. If a numeric budget is provided, explicitly state\n"
         "whether the estimated total is within, at, or over that budget."),
        ("human",
         "Destination: {destination}\n"
         "Dates: {travel_dates}\n"
         "Budget: {budget}\n"
         "Itinerary: {itinerary}"),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "destination": state["destination"],
        "travel_dates": state["travel_dates"],
        "budget": state["budget"],
        "itinerary": state["itinerary"],
    })
    return {"budget_estimate": response.content}


# ============================================================================
# LANGGRAPH WORKFLOW DEFINITION
# ============================================================================

def build_graph():
    """
    Build the LangGraph workflow connecting all agents.
    
    Workflow:
        START → Planner → Research → Itinerary Builder → Budget Estimator → END
    
    Each agent processes the shared TravelState and passes it to the next agent.
    """
    graph = StateGraph(TravelState)
    
    # Add nodes (agents)
    graph.add_node("planner", planner_node)
    graph.add_node("research", research_node)
    graph.add_node("itinerary", itinerary_node)
    graph.add_node("budget", budget_node)
    
    # Define workflow edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "itinerary")
    graph.add_edge("itinerary", "budget")
    graph.add_edge("budget", END)
    
    return graph.compile()


# ============================================================================
# EXECUTION FUNCTION
# ============================================================================

def execute_travel_planning(user_input: str) -> TravelState:
    """
    Execute the multi-agent travel planning workflow.
    
    Args:
        user_input: User's travel request in natural language
    
    Returns:
        TravelState: Final state with all fields populated by agents
    """
    initial_state: TravelState = {
        "user_input": user_input,
        "destination": "",
        "travel_dates": "",
        "budget": "",
        "preferences": "",
        "research_notes": "",
        "itinerary": "",
        "budget_estimate": "",
    }
    
    graph = build_graph()
    final_state = graph.invoke(initial_state)
    return final_state


# ============================================================================
# CLI MODE
# ============================================================================

def print_travel_plan(state: TravelState) -> None:
    """Print travel plan to console (CLI mode)"""
    print("\n" + "="*80)
    print("TRAVEL PLAN GENERATED")
    print("="*80)
    
    print("\n📍 DESTINATION")
    print("-" * 80)
    print(state["destination"] if state["destination"] else "Not determined")
    
    print("\n📅 TRAVEL DATES")
    print("-" * 80)
    print(state["travel_dates"] if state["travel_dates"] else "Not specified")
    
    print("\n💰 BUDGET")
    print("-" * 80)
    print(state["budget"] if state["budget"] else "Not specified")
    
    print("\n🎯 PREFERENCES")
    print("-" * 80)
    print(state["preferences"] if state["preferences"] else "Not specified")
    
    print("\n📝 ITINERARY")
    print("-" * 80)
    print(state["itinerary"] if state["itinerary"] else "No itinerary generated")
    
    print("\n💵 BUDGET ESTIMATE")
    print("-" * 80)
    print(state["budget_estimate"] if state["budget_estimate"] else "No estimate generated")
    
    print("\n" + "="*80)


def main() -> None:
    """Main function for CLI mode"""
    print("="*80)
    print("MULTI-AGENT TRAVEL PLANNER")
    print("="*80)
    print("\nThis system uses 4 AI agents to create your perfect travel plan:")
    print("  1. Planner Agent - Extracts travel details")
    print("  2. Research Agent - Finds attractions and tips")
    print("  3. Itinerary Builder - Creates day-by-day plans")
    print("  4. Budget Estimator - Analyzes costs")
    print("\n" + "="*80 + "\n")
    
    # Get user input
    user_input = ""
    while not user_input.strip():
        user_input = input("Enter your travel request: ")
    
    print("\n🔄 Processing your request through 4 specialized agents...")
    print("   This may take 15-30 seconds...\n")
    
    # Execute workflow
    final_state = execute_travel_planning(user_input)
    
    # Display results
    print_travel_plan(final_state)


# ============================================================================
# STREAMLIT MODE
# ============================================================================

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="AI Travel Planner",
        page_icon="🌍",
        layout="wide"
    )

    # =========================
    # DARK MODERN UI
    # =========================
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        color: #9aa4b2;
        margin-bottom: 2rem;
    }

    .card {
        background: #161b22;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #30363d;
    }

    .highlight {
        color: #58a6ff;
        font-weight: bold;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: #c9d1d9;
    }

    textarea {
        background-color: #0d1117 !important;
        color: white !important;
        border-radius: 10px !important;
        border: 1px solid #30363d !important;
    }

    div.stButton > button {
        background-color: #238636;
        color: white;
        border-radius: 10px;
        font-weight: 600;
        border: none;
        padding: 0.6rem;
    }

    div.stButton > button:hover {
        background-color: #2ea043;
    }
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # HEADER
    # =========================
    st.markdown('<div class="main-title">🌍 AI Travel Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Multi-Agent System powered by LangGraph</div>', unsafe_allow_html=True)

    # =========================
    # SIDEBAR
    # =========================
    with st.sidebar:
        st.markdown("## ⚙️ System Overview")

        st.markdown("""
        **🧠 Planner Agent**  
        Extracts travel details  

        **🔎 Research Agent**  
        Finds attractions & insights  

        **📅 Itinerary Agent**  
        Builds daily plan  

        **💰 Budget Agent**  
        Estimates expenses  
        """)

        st.markdown("---")

        st.markdown("### 🛠 Tech Stack")
        st.markdown("""
        - LangGraph  
        - LangChain  
        - Groq LLM  
        - Streamlit  
        """)

    # =========================
    # INPUT
    # =========================
    st.markdown('<div class="section-title">📝 Describe Your Trip</div>', unsafe_allow_html=True)

    user_input = st.text_area(
        "",
        height=120,
        placeholder="Example: Trip to Japan for 7 days in March, budget $3000, love food & temples"
    )

    generate_button = st.button("🚀 Generate Plan", use_container_width=True)

    # =========================
    # PROCESS
    # =========================
    if generate_button:
        if not user_input.strip():
            st.warning("Enter something first. Empty input = garbage output.")
        else:
            try:
                with st.spinner("Agents working..."):
                    final_state = execute_travel_planning(user_input)

                st.success("Plan Ready")

                # =========================
                # TABS
                # =========================
                tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Itinerary", "Budget", "Research"])

                # -------- OVERVIEW --------
                with tab1:
                    st.markdown('<div class="section-title">📍 Overview</div>', unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"""
                        <div class="card">
                            <div class="highlight">Destination</div>
                            {final_state["destination"] or "N/A"}
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="card">
                            <div class="highlight">Budget</div>
                            {final_state["budget"] or "N/A"}
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="card">
                            <div class="highlight">Dates</div>
                            {final_state["travel_dates"] or "N/A"}
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="card">
                            <div class="highlight">Preferences</div>
                            {final_state["preferences"] or "N/A"}
                        </div>
                        """, unsafe_allow_html=True)

                # -------- ITINERARY --------
                with tab2:
                    st.markdown('<div class="section-title">📝 Itinerary</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="card">
                    {final_state["itinerary"] or "No itinerary generated"}
                    </div>
                    """, unsafe_allow_html=True)

                # -------- BUDGET --------
                with tab3:
                    st.markdown('<div class="section-title">💰 Budget Analysis</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="card">
                    {final_state["budget_estimate"] or "No budget estimate"}
                    </div>
                    """, unsafe_allow_html=True)

                # -------- RESEARCH --------
                with tab4:
                    st.markdown('<div class="section-title">🔍 Research</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="card">
                    {final_state["research_notes"] or "No research"}
                    </div>
                    """, unsafe_allow_html=True)

                # =========================
                # DOWNLOAD
                # =========================
                st.markdown("---")

                travel_plan_text = f"""
Destination: {final_state['destination']}
Dates: {final_state['travel_dates']}
Budget: {final_state['budget']}
Preferences: {final_state['preferences']}

ITINERARY:
{final_state['itinerary']}

BUDGET:
{final_state['budget_estimate']}

RESEARCH:
{final_state['research_notes']}
"""

                st.download_button(
                    label="📥 Download Plan",
                    data=travel_plan_text,
                    file_name="travel_plan.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.error("Check your GROQ_API_KEY or backend logic")
# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        import streamlit as st
        # If streamlit is imported and we're in streamlit context
        if hasattr(st, 'runtime') and st.runtime.exists():
            run_streamlit_app()
        else:
            # Streamlit installed but not running in streamlit context
            main()
    except ImportError:
        # Streamlit not installed, run CLI mode
        main()