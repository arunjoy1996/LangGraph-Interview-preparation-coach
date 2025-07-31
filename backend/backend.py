from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Literal
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
import random
import json
import os
import threading
import uuid

os.environ["GROQ_API_KEY"] = ""
model = ChatGroq(model_name="llama3-70b-8192", temperature=0.0)

try:
    with open("questions.json", "r") as f:
        QUESTION_BANK = json.load(f)
except FileNotFoundError:
    raise Exception("questions.json not found")
except json.JSONDecodeError:
    raise Exception("Invalid questions.json format")

# Thread-safe session store
SESSIONS: Dict[str, Dict] = {}
SESSIONS_LOCK = threading.Lock()


class InterviewState(TypedDict):
    messages: List[BaseMessage]
    current_question: str
    used_questions: List[str]
    evaluations: List[str]
    feedback: List[str]
    round: int
    max_rounds: int
    difficulty: Literal["easy", "medium", "hard"]
    category: Literal["behavioral", "technical"]
    summary: str
    user_response: str  



def flatten_messages(messages: List[BaseMessage]) -> str:
    role_map = {"human": "User", "ai": "Assistant", "system": "System"}
    valid_types = role_map.keys()
    result = []
    for msg in messages:
        if msg.type not in valid_types:
            print(f"Warning: Unknown message type {msg.type}")
        result.append(f"{role_map.get(msg.type, 'Unknown')}: {msg.content}")
    return "\n".join(result)



def select_question(state: InterviewState) -> InterviewState:
    """Select and present a new question"""
    difficulty = state.get("difficulty", "medium")
    category = state.get("category", "behavioral")
    pool = QUESTION_BANK.get(category, {}).get(difficulty, [])
    used_questions = state.get("used_questions", [])

    available_questions = [q for q in pool if q not in used_questions]
    if not available_questions:
        question = "No more questions available."
    else:
        # Use round number as seed for consistent selection
        random.seed(f"{state.get('round', 0)}-{len(used_questions)}")
        question = random.choice(available_questions)

    return {
        **state,
        "current_question": question,
        "used_questions": used_questions + [question] if question != "No more questions available." else used_questions,
        "messages": state["messages"] + [AIMessage(content=f"Question {state.get('round', 0) + 1}: {question}")]
    }

def wait_for_user_input(state: InterviewState) -> InterviewState:
    """This node waits for user input - execution will be interrupted here"""
    # This is where the graph will pause - user input will be injected via API
    return state

def process_user_response(state: InterviewState) -> InterviewState:
    """Process the user's response after it's been injected"""
    user_response = state.get("user_response", "")
    if user_response:
        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content=user_response)]
        }
    return state

def evaluate_response(state: InterviewState) -> InterviewState:
    """Evaluate the user's response"""
    messages_str = flatten_messages(state["messages"][-2:])  # Last question and answer
    prompt = f"""
    Based on this interview exchange:
    {messages_str}
    
    Evaluate the user's response to the interview question. Comment on which part is missing or weak. Be concise and specific.
    """
    
    try:
        evaluation = model.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        evaluation = f"Error evaluating response: {str(e)}"
    
    return {
        **state,
        "evaluations": state["evaluations"] + [evaluation]
    }

def give_feedback(state: InterviewState) -> InterviewState:
    """Give constructive feedback based on evaluation"""
    last_evaluation = state["evaluations"][-1] if state["evaluations"] else ""
    prompt = f"""
    Based on this evaluation: {last_evaluation}
    
    Give friendly, constructive feedback to the candidate. Mention one specific area to work on and one thing they did well.
    Keep it encouraging but actionable.
    """
    
    try:
        feedback = model.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        feedback = f"Error generating feedback: {str(e)}"
    
    return {
        **state,
        "feedback": state["feedback"] + [feedback],
        "round": state["round"] + 1,
        "user_response": ""  # Clear user response for next round
    }

def check_continue(state: InterviewState) -> str:
    """Determine whether to continue with more questions or summarize"""
    if state["round"] >= state["max_rounds"]:
        return "summarize_interview"
    else:
        return "select_question"

def summarize_interview(state: InterviewState) -> InterviewState:
    """Generate final summary of the interview"""
    all_evaluations = "\n".join(state["evaluations"])
    all_feedback = "\n".join(state["feedback"])
    
    prompt = f"""
    You are an interview coach. Based on the following evaluations and feedback from a {state['max_rounds']}-question interview:
    
    EVALUATIONS:
    {all_evaluations}
    
    FEEDBACK:
    {all_feedback}
    
    Provide a comprehensive summary of the candidate's overall performance. Include:
    1. Key strengths demonstrated
    2. Main areas for improvement
    3. Specific recommendations for interview preparation
    4. Overall assessment
    
    Be encouraging but honest in your assessment.
    """
    
    try:
        summary = model.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        summary = f"Error generating summary: {str(e)}"
    
    return {
        **state,
        "summary": summary
    }


builder = StateGraph(InterviewState)

# Add nodes
builder.add_node("select_question", select_question)
builder.add_node("wait_for_user_input", wait_for_user_input)
builder.add_node("process_user_response", process_user_response)
builder.add_node("evaluate_response", evaluate_response)
builder.add_node("give_feedback", give_feedback)
builder.add_node("summarize_interview", summarize_interview)

# Set entry point
builder.set_entry_point("select_question")

# Add edges
builder.add_edge("select_question", "wait_for_user_input")
builder.add_edge("wait_for_user_input", "process_user_response")
builder.add_edge("process_user_response", "evaluate_response")
builder.add_edge("evaluate_response", "give_feedback")
builder.add_conditional_edges("give_feedback", check_continue)
builder.add_edge("summarize_interview", END)

# Compile graph with interrupts
graph = builder.compile(
    interrupt_before=["wait_for_user_input"],  # Pause before waiting for user input
    checkpointer=MemorySaver()
)



app = FastAPI()

class StartInterviewRequest(BaseModel):
    session_id: str
    rounds: int = 3
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    category: Literal["behavioral", "technical"] = "behavioral"

class AnswerRequest(BaseModel):
    session_id: str
    user_message: str

@app.post("/start")
async def start_interview(req: StartInterviewRequest):
    """Start a new interview session"""
    if req.rounds <= 0:
        raise HTTPException(status_code=400, detail="max_rounds must be positive")
    
    with SESSIONS_LOCK:
        if req.session_id in SESSIONS:
            raise HTTPException(status_code=400, detail="Session ID already in use")
    
    # Initialize state
    initial_state: InterviewState = {
        "messages": [],
        "current_question": "",
        "used_questions": [],
        "evaluations": [],
        "feedback": [],
        "round": 0,
        "max_rounds": req.rounds,
        "difficulty": req.difficulty,
        "category": req.category,
        "summary": "",
        "user_response": ""
    }

    # Start the graph - it will pause at wait_for_user_input
    config = {"configurable": {"thread_id": req.session_id}}
    
    try:
        # This will execute until the first interrupt (wait_for_user_input)
        result = graph.invoke(initial_state, config)
        
        # Get the current state from the checkpoint
        state_snapshot = graph.get_state(config)
        current_state = state_snapshot.values
        
        with SESSIONS_LOCK:
            SESSIONS[req.session_id] = {"initialized": True}
        
        return {
            "question": current_state["current_question"],
            "round": current_state["round"] + 1
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting interview: {str(e)}")

@app.post("/answer")
async def answer(req: AnswerRequest):
    """Submit an answer and continue the interview"""
    with SESSIONS_LOCK:
        if req.session_id not in SESSIONS:
            raise HTTPException(status_code=404, detail="Invalid session_id")
    
    config = {"configurable": {"thread_id": req.session_id}}
    
    try:
        # Get current state
        state_snapshot = graph.get_state(config)
        if not state_snapshot:
            raise HTTPException(status_code=400, detail="No active session found")
        
        current_state = state_snapshot.values
        
        # Check if we're at the right point to accept input
        if state_snapshot.next != ("wait_for_user_input",):
            raise HTTPException(status_code=400, detail="Not ready for user input")
        
        # Update state with user response
        updated_state = {
            **current_state,
            "user_response": req.user_message
        }
        
        # Update the state in the checkpoint
        graph.update_state(config, updated_state)
        
        # Continue execution - this will process the response, evaluate, give feedback
        # and either continue to next question or summarize
        graph.invoke(None, config)
        
        # Get the final state after processing
        final_state_snapshot = graph.get_state(config)
        final_state = final_state_snapshot.values
        
        # Check if interview is complete
        is_done = final_state["round"] >= final_state["max_rounds"]
        
        response = {
            "evaluations": final_state["evaluations"][-1] if final_state["evaluations"] else "",
            "feedback": final_state["feedback"][-1] if final_state["feedback"] else "",
            "done": is_done
        }
        
        if is_done:
            response["summary"] = final_state.get("summary", "")
        else:
            response["question"] = final_state["current_question"]
            response["round"] = final_state["round"] + 1
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing answer: {str(e)}")

@app.get("/status")
async def get_status(session_id: str):
    """Get current interview status"""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        state_snapshot = graph.get_state(config)
        if not state_snapshot:
            raise HTTPException(status_code=404, detail="Session not found")
        
        state = state_snapshot.values
        return {
            "current_question": state.get("current_question", ""),
            "round": state.get("round", 0),
            "max_rounds": state.get("max_rounds", 0),
            "done": state.get("round", 0) >= state.get("max_rounds", 0),
            "waiting_for_input": state_snapshot.next == ("wait_for_user_input",) if state_snapshot.next else False
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.get("/summary")
async def get_summary(session_id: str):
    """Get final interview summary"""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        state_snapshot = graph.get_state(config)
        if not state_snapshot:
            raise HTTPException(status_code=404, detail="Session not found")
            
        state = state_snapshot.values
        return {"summary": state.get("summary", "")}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")

@app.post("/reset")
async def reset_session(session_id: str):
    """Reset an interview session"""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        with SESSIONS_LOCK:
            if session_id in SESSIONS:
                del SESSIONS[session_id]
        
        # Clear the checkpoint state
        graph.get_state(config)  # This ensures the checkpoint exists
        return {"message": "Session reset successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting session: {str(e)}")
