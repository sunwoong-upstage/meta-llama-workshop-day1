#!/usr/bin/env python3
"""
Real-world Customer Support Agent Demo with Interactive Flow
"""

# ============================================================================
# IMPORTS AND ENVIRONMENT SETUP
# ============================================================================

import os
import json
import uuid
import csv
import numpy as np
import faiss
from typing import TypedDict, List, Dict, Any, Annotated, Literal
from datetime import datetime
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_upstage import ChatUpstage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
# InMemorySaver not needed for langgraph dev - persistence handled automatically

# External APIs
from openai import OpenAI
from tavily import TavilyClient
from pydantic import BaseModel

# Load environment variables
load_dotenv()
print("✅ Environment loaded")

# Set up client (Upstage API)
client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1"
)
# ============================================================================
# DATA SETUP AND CONFIGURATION
# ============================================================================

# Load customer data from CSV
def load_customer_data(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Load customer data from CSV file."""
    customers = {}
    

    with open(csv_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            if not row.get('customer_id') or not row.get('customer_id', '').strip():
                continue
                
            customers[row['customer_id']] = {
                'customer_id': row['customer_id'],
                'name': row['name'],
                'email': row['email'],
                'plan': row['plan'],
                'subscription_status': row['subscription_status'],
                'last_login': row['last_login'],
                'account_age_days': int(row['account_age_days']),
                'previous_issues': int(row['previous_issues'])
            }
    print(f"✅ Loaded {len(customers)} customers from CSV file")
    return customers

customers = load_customer_data("data/customer_data.csv")

# Specialists data
specialists = {
    "Technical": {
        "specialist": "Alex Chen",
        "email": "alex.chen@ourcompany.com",
        "expertise": ["API Integration", "System Architecture", "Performance Issues"],
        "response_time": "2-4 hours"
    },
    "Billing": {
        "specialist": "Maria Rodriguez",
        "email": "maria.rodriguez@ourcompany.com",
        "expertise": ["Payment Processing", "Refunds", "Subscription Management"],
        "response_time": "1-2 hours"
    },
    "Account": {
        "specialist": "James Wilson",
        "email": "james.wilson@ourcompany.com",
        "expertise": ["Account Management", "Security", "Access Issues"],
        "response_time": "1-3 hours"
    },
    "General": {
        "specialist": "Sarah Thompson",
        "email": "sarah.thompson@ourcompany.com",
        "expertise": ["General Inquiries", "Feature Requests", "Feedback"],
        "response_time": "4-8 hours"
    },
    "Urgent": {
        "specialist": "Emergency Team",
        "email": "emergency@ourcompany.com",
        "expertise": ["Critical Issues", "System Outages", "Security Incidents"],
        "response_time": "15-30 minutes"
    }
}


# Knowledge base documents
knowledge_base_documents = [
    Document(
        page_content="To reset your password, go to the login page and click 'Forgot Password'. Enter your email address and check your inbox for reset instructions. The reset link expires in 24 hours.",
        metadata={"category": "Account", "topic": "Password Reset", "priority": "High"}
    ),
    Document(
        page_content="For billing issues, you can view your invoices in the Billing section of your account dashboard. Payment methods can be updated under Account Settings > Payment Information. Refunds are processed within 5-7 business days.",
        metadata={"category": "Billing", "topic": "Payment Management", "priority": "High"}
    ),
    Document(
        page_content="Our API rate limits are 1000 requests per hour for Basic plans, 5000 for Premium, and 10000 for Enterprise. If you exceed these limits, you'll receive a 429 status code. Consider upgrading your plan for higher limits.",
        metadata={"category": "Technical", "topic": "API Limits", "priority": "Medium"}
    ),
    Document(
        page_content="To integrate our API, use the base URL https://api.ourcompany.com/v1. Authentication requires an API key in the Authorization header. Example: Authorization: Bearer your_api_key_here",
        metadata={"category": "Technical", "topic": "API Integration", "priority": "High"}
    ),
    Document(
        page_content="Data export is available for all plans. Go to Account Settings > Data Export to request your data. The export will be emailed to you within 24 hours and includes all your account data in JSON format.",
        metadata={"category": "Account", "topic": "Data Export", "priority": "Medium"}
    ),
    Document(
        page_content="Two-factor authentication (2FA) can be enabled in Security Settings. We support SMS, email, and authenticator apps. 2FA is required for Enterprise accounts and recommended for all users.",
        metadata={"category": "Account", "topic": "Security", "priority": "High"}
    ),
    Document(
        page_content="Subscription upgrades take effect immediately. Downgrades take effect at the next billing cycle. You can change your plan anytime in Account Settings > Subscription Management.",
        metadata={"category": "Billing", "topic": "Plan Changes", "priority": "Medium"}
    ),
    Document(
        page_content="Our service status page is available at status.ourcompany.com. We post real-time updates about any service disruptions, maintenance windows, or performance issues.",
        metadata={"category": "Technical", "topic": "Service Status", "priority": "High"}
    ),
    Document(
        page_content="For enterprise customers, we offer dedicated support channels including phone support, dedicated account managers, and custom SLA agreements. Contact sales@ourcompany.com for more information.",
        metadata={"category": "General", "topic": "Enterprise Support", "priority": "Low"}
    ),
    Document(
        page_content="Webhook configuration is available in the Developer section. You can set up webhooks for events like payment success, user registration, and data updates. Webhook URLs must use HTTPS.",
        metadata={"category": "Technical", "topic": "Webhooks", "priority": "Medium"}
    )
]

# Create FAISS index for knowledge base
texts = [doc.page_content for doc in knowledge_base_documents]

client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1"
)

def _get_embeddings(client: OpenAI, texts: List[str], mode: str = "query") -> List[List[float]]:
    model = "embedding-query" if mode == "query" else "embedding-passage"
    response = client.embeddings.create(model=model, input=texts)
    return [embedding.embedding for embedding in response.data]

embeddings = _get_embeddings(client, texts, mode="passage")
document_embeddings = np.array(embeddings).astype(np.float32)

dimension = len(embeddings[0])
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(document_embeddings)
index.add(document_embeddings)

print(f"✅ Knowledge base index created with {len(texts)} documents (dimension: {dimension})")
    
# ============================================================================
# GRAPH STATE DEFINITION
# ============================================================================

class GraphState(TypedDict):
    customer_question: str
    customer_id: str
    customer_context: str
    knowledge_base_results: List[Dict[str, Any]]
    web_search_raw_results: List[Dict[str, Any]]
    specialist_info: Dict[str, Any]
    messages: Annotated[List[BaseMessage], add_messages]

# ============================================================================
# CLASSIFICATION SCHEMA
# ============================================================================

class InquiryCategorySchema(BaseModel):
    category: Literal["Technical", "Billing", "Account", "General", "Urgent"]

def classify_inquiry(customer_question: str) -> str:
    """Use LLM with structured output to classify customer inquiry."""
    llm = ChatUpstage(model="solar-pro2", temperature=0)
    llm_structured = llm.with_structured_output(InquiryCategorySchema)
    
    classification_prompt = f"""
    Classify the following customer inquiry into one of these categories:
    - Technical: Issues related to API, software, technical problems, bugs, system errors
    - Billing: Payment issues, subscription problems, billing questions, refunds
    - Account: Account management, login issues, password reset, profile changes
    - General: General questions, information requests, non-specific inquiries
    - Urgent: Critical issues requiring immediate attention, system outages, security breaches
    
    Customer Question: "{customer_question}"
    
    Classify this inquiry into the most appropriate category.
    """
    
    result = llm_structured.invoke(classification_prompt)
    return result.category


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

@tool
def search_knowledge_base(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Search the internal knowledge base using vector similarity."""
    if query is None or not str(query).strip():
        return [{"error": "query is required"}]

    client = OpenAI(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1"
    )
    q_emb = _get_embeddings(client, [query], mode="query")[0]
    q_vec = np.array([q_emb]).astype(np.float32)
    faiss.normalize_L2(q_vec)

    scores, idxs = index.search(q_vec, k=max(1, int(top_k)))

    results = []
    for i, (score, idx) in enumerate(zip(scores[0], idxs[0])):
        if idx < len(knowledge_base_documents):
            doc = knowledge_base_documents[idx]
        
            result = {
                "content": doc.page_content,
                "score": float(score),
                "topic": doc.metadata.get('topic', 'General'),
                "category": doc.metadata.get('category', 'N/A'),
                "index": int(idx),
                "rank": i + 1
            }
            results.append(result)

    if not results:
        return [{"message": "No relevant information found in the knowledge base."}]
    
    return results



def _rewrite_query_for_search(query: str) -> str:
    """Use LLM to rewrite customer question into optimal search query.
    
    Args:
        query: Original search query
        
    Returns:
        LLM-optimized search query
    """
    try:        
        current_time = datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")
        
        rewrite_prompt = f"""
        <role>
        You are an expert search query optimizer. Your task is to transform verbose, conversational user questions into concise, keyword-driven search queries that will yield the best possible results from a web search engine.
        </role>

        <instructions>
        1.  **Identify Core Intent:** Analyze the user's question to understand their fundamental goal.
        2.  **Extract Key Entities:** Pull out essential keywords, names, locations, constraints (like budget or time), and concepts.
        3.  **Remove Filler:** Discard conversational fluff (e.g., "I was wondering", "can you help me", "please", "I think").
        4.  **Synthesize Keywords:** Combine the extracted entities into a logical, concise search string.
        5.  **Keep it Brief:** The final query should ideally be under 10 words.
        6.  **IMPORTANT:** Return ONLY the search query, no explanations, no tags, no thinking process.
        7.  **Time Context:** If user asks time-specific question, use current time information (current date: {current_date}).
        </instructions>

        <example>
        <user_question>
        I'm thinking of going to Europe this winter, maybe for like a week. I'm on a budget but I still want to see some cool historical stuff. Can you give me some recommendations?
        </user_question>
        <rewritten_query>
        best budget winter destinations Europe historical sites one week
        </rewritten_query>
        </example>

        <user_question>
        {query}
        </user_question>

        <output_format>
        Provide ONLY the rewritten search query as a single line of text. Do not add any introductory phrases like "Here is the query:".
        And also DO NOT PROVIDE THE THINKING STEPS, just provide the rewritten query.
        </output_format>

        <rewritten_query>
        """
    
        response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "search_query_suggestions",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "rewritten_query": {
                        "type": "string",
                        "description": "The most relevant search query for this inquiry"
                    }
                },
                "required": ["rewritten_query"]
            }
        }
    }
    
        response = client.chat.completions.create(
            model="solar-pro2",
            messages=[{"role": "system", "content": rewrite_prompt}],
            # max_tokens=100,
            response_format=response_format
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Return the primary query as the main result
        return result["rewritten_query"]
        
    except Exception as e:
        print(f"   ⚠️ LLM query rewrite failed: {str(e)}, using original query")
        return query


@tool
def web_search(query: str, max_results: int = 3, rewrite_mode: bool = True) -> List[Dict[str, Any]]:
    """Search the web for relevant information about a customer query.
    
    This tool searches the internet to find current information that can help
    answer customer questions, especially for technical issues or general topics
    not covered in the internal knowledge base.
    
    Args:
        query: The customer's question or search query
        max_results: Maximum number of search results to return (default: 3)
        rewrite_mode: Whether to use LLM to optimize the search query (default: True)
        
    Returns:
        List of dictionaries containing structured web search results with scores and metadata
    """
    try:
        # Initialize Tavily client
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # Optionally rewrite query for better search results
        search_query = query
        if rewrite_mode:
            search_query = _rewrite_query_for_search(query)
        print(f"Search with the query: {search_query}")
        
        # Search the web
        response = tavily_client.search(
            query=search_query,
            search_depth="basic",
            max_results=max_results,
            include_domains=[],
            exclude_domains=[]
        )
        
        if not response.get('results'):
            return [{"message": "No relevant web search results found."}]
        
        # Return structured results
        results = []
        for i, result in enumerate(response['results'], 1):
            structured_result = {
                "title": result.get('title', 'Untitled'),
                "content": result.get('content', 'No content available'),
                "url": result.get('url', 'No URL'),
                "score": float(result.get('score', 0.0)),
                "rank": i,
                "query": search_query
            }
            results.append(structured_result)
        
        return results
        
    except Exception as e:
        return [{"error": f"Web search error: {str(e)}"}]


def fetch_customer_data(customer_id: str) -> str:
    """Fetch customer data from CRM system."""
    try:
        customer_data = customers.get(customer_id, {})
        
        if not customer_data:
            return "No customer data found for this ID."
        
        context_parts = []
        if customer_data.get('name'):
            context_parts.append(f"Name: {customer_data['name']}")
        if customer_data.get('email'):
            context_parts.append(f"Email: {customer_data['email']}")
        if customer_data.get('plan'):
            context_parts.append(f"Plan: {customer_data['plan']}")
        if customer_data.get('subscription_status'):
            context_parts.append(f"Status: {customer_data['subscription_status']}")
        if customer_data.get('account_age_days'):
            context_parts.append(f"Account Age: {customer_data['account_age_days']} days")
        if customer_data.get('previous_issues'):
            context_parts.append(f"Previous Issues: {customer_data['previous_issues']}")
        
        customer_context = "Customer Information:\n" + "\n".join(context_parts)
        print(f"✅ Customer context retrieved for {customer_id}")
        return customer_context
        
    except Exception as e:
        return f"❌ Customer data fetch error: {str(e)}"

# ============================================================================
# MODEL SETUP WITH TOOLS
# ============================================================================

tools = [search_knowledge_base, web_search]
model = ChatUpstage(model="solar-pro2", temperature=0)
model_with_tools = model.bind_tools(tools)

print("✅ Model with tools created")

# ============================================================================
# TOOL NODE IMPLEMENTATION
# ============================================================================

def tool_node(state: GraphState) -> Dict[str, Any]:
    """Execute tools and update state with structured results."""
    from langchain_core.messages import ToolMessage
    
    outputs = []
    update = {"messages": []}
    
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = None
        for tool in tools:
            if tool.name == tool_call["name"]:
                tool_result = tool.invoke(tool_call["args"])
                break
        
        # Create proper ToolMessage
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result) if tool_result else "Tool not found",
                tool_call_id=tool_call["id"],
            )
        )
        
        if tool_call["name"] == "search_knowledge_base":
            kb_results = tool_result if isinstance(tool_result, list) else []
            update["knowledge_base_results"] = kb_results
            print(f"📚 KB search completed with {len(kb_results)} results")
        
        if tool_call["name"] == "web_search":
            web_results = tool_result if isinstance(tool_result, list) else []
            update["web_search_raw_results"] = web_results
            print(f"🌐 Web search completed with {len(web_results)} results")
    
    update["messages"] = outputs
    return update

# ============================================================================
# GRAPH NODE FUNCTIONS
# ============================================================================

def get_customer_id(state: GraphState) -> Dict[str, Any]:
    """Get customer ID from user."""
    customer_id = interrupt("🆔 Welcome to Customer Support! Can you please enter your Customer ID (CUSTXXX format)?")
    
    # Validate customer ID format
    if not customer_id or not customer_id.startswith("CUST"):
        customer_id = interrupt(f"❌ Invalid format '{customer_id}'. Please enter your Customer ID in CUSTXXX format:")
    
    # Check if customer has changed - if so, reset search results
    previous_customer_id = state.get("customer_id", "")
    update_dict = {"customer_id": customer_id}
    
    if previous_customer_id and previous_customer_id != customer_id:
        print(f"🔄 Customer changed from {previous_customer_id} to {customer_id} - resetting search results")
        update_dict.update({
            "knowledge_base_results": [],
            "web_search_raw_results": [],
            "specialist_info": {},
            "customer_question": ""  # Reset previous question too
        })
    elif not previous_customer_id:
        print(f"🆕 New session for customer {customer_id}")
        # Initialize empty results for new session
        update_dict.update({
            "knowledge_base_results": [],
            "web_search_raw_results": [],
            "specialist_info": {},
            "customer_question": ""
        })
    else:
        print(f"🔄 Same customer {customer_id} - keeping existing context but resetting search results for new question")
        # Even for same customer, reset search results for new question
        update_dict.update({
            "knowledge_base_results": [],
            "web_search_raw_results": [],
            "specialist_info": {},
            "customer_question": ""
        })
    
    # Fetch customer context
    customer_context = fetch_customer_data(customer_id)
    update_dict["customer_context"] = customer_context
    
    print(f"✅ Customer context retrieved for {customer_id}")
    print(f"✅ Customer context: {customer_context}")
    
    return update_dict

def get_customer_question(state: GraphState) -> Dict[str, Any]:
    """Get customer question from user."""
    customer_question = interrupt("❓ Do you need any help? Please describe your issue or question:")
    
    if not customer_question or len(customer_question.strip()) < 5:
        customer_question = interrupt("❓ Please provide more details about your issue:")
    
    return {"customer_question": customer_question}

def llm_node(state: GraphState) -> Dict[str, Any]:
    """Main LLM node that processes customer requests and calls tools."""
    customer_id = state.get("customer_id", "Unknown")
    customer_question = state.get("customer_question", "")
    customer_context = state.get("customer_context", "No customer context available")
    print(f"🤖 Processing question for {customer_id}: {customer_question[:100]}...")
    
    # Check if we have tool results by looking at state fields directly
    kb_results = state.get("knowledge_base_results", [])
    web_results = state.get("web_search_raw_results", [])
    has_tool_results = len(kb_results) > 0 or len(web_results) > 0
    
    if has_tool_results:
        print("We have tool results")
        
        # Get customer context
        
        
        # Format knowledge base results
        kb_context = ""
        if kb_results:
            kb_formatted = []
            for result in kb_results:
                if isinstance(result, dict) and 'content' in result:
                    topic = result.get('topic', 'Unknown')
                    category = result.get('category', 'Unknown')
                    content = result.get('content', '')
                    score = result.get('score', 0.0)
                    kb_formatted.append(f"- [{category} - {topic}] (Score: {score:.1%}): {content}")
            kb_context = "\n".join(kb_formatted)
        
        # Format web search results
        web_context = ""
        if web_results:
            web_formatted = []
            for result in web_results:
                if isinstance(result, dict) and 'content' in result:
                    title = result.get('title', 'Untitled')
                    content = result.get('content', '')
                    url = result.get('url', '')
                    score = result.get('score', 0.0)
                    web_formatted.append(f"- [{title}] (Score: {score:.1%}): {content}\n  Source: {url}")
            web_context = "\n".join(web_formatted)
        
        # Construct comprehensive system prompt with all context
        system_prompt = f"""You are a Customer Service Assistant helping customer {customer_id}.

CUSTOMER CONTEXT:
{customer_context}

AVAILABLE INFORMATION:"""
        
        if kb_context:
            system_prompt += f"""

KNOWLEDGE BASE RESULTS:
{kb_context}"""
        
        if web_context:
            system_prompt += f"""

WEB SEARCH RESULTS:
{web_context}"""
        
        system_prompt += f"""

CUSTOMER QUESTION: {customer_question}

INSTRUCTIONS:
Use the above information to provide a comprehensive, helpful answer to the customer's question. 
Reference the relevant information from the search results and consider the customer's context.
Be specific and cite the sources when appropriate.
Do NOT call any tools - provide your final answer based on the available information."""

        system = SystemMessage(system_prompt)
        
    else:
        print("First pass: LLM decides whether to use tools or answer directly")
        # First pass: LLM decides whether to use tools or answer directly
        system = SystemMessage(
            f"You are a Customer Service Assistant helping customer {customer_id}.\n"
            f"Customer context: {customer_context}\n"
            f"Available tools: search_knowledge_base, web_search.\n\n"
            f"Except the case that the question is really general, you are recommended to use the tools to answer the question."
            f"Customer question: '{customer_question}'\n\n"
            f"Instructions:\n"
            f"1. If you can fully answer this question with your knowledge, provide a complete, helpful response.\n"
            f"2. If you need specific information, call the appropriate tools to gather information.\n"
            f"3. If you cannot help or the request is beyond your capabilities, clearly state: 'I cannot help with this and need to escalate this to a specialist.'"
        )
    
    print(f"🔧 System message: {system.content[:200]}...")
    print(f"📊 Tool results available: KB={len(kb_results)}, Web={len(web_results)}")
    
    # For tool results case, we have all context in system prompt, so just pass customer question
    # For first pass, we need the model to potentially call tools
    response = model_with_tools.invoke([system, HumanMessage(content=customer_question)])
    
    return {
        "messages": [response]
    }


def escalate_to_specialist(state: GraphState) -> Dict[str, Any]:
    """Classify inquiry, assign specialist, and notify customer."""
    print("🚨 Escalating to specialist")
    
    customer_question = state.get("customer_question", "")
    
    # Classify inquiry for specialist assignment
    inquiry_category = classify_inquiry(customer_question)
    specialist_info = specialists.get(inquiry_category, specialists["General"])
    
    print(f"🏷️ Classified as: {inquiry_category}")
    print(f"📋 Escalating to: {specialist_info['specialist']}")
    
    # Simulate sending to specialist
    print(f"   ✉️ Escalation sent to {specialist_info.get('email', 'unknown')}")
    
    # Generate customer notification
    escalation_message = HumanMessage(
        content=f"I apologize, but I wasn't able to find sufficient information to help you with your inquiry. "
                f"I'm escalating your case to our specialist team.\n\n"
                f"🏷️ **Assigned Specialist**: {specialist_info.get('specialist', 'Unknown')}\n"
                f"📧 **Email**: {specialist_info.get('email', 'Unknown')}\n"
                f"🎯 **Expertise**: {', '.join(specialist_info.get('expertise', ['General Support']))}\n"
                f"⏱️ **Expected Response Time**: {specialist_info.get('response_time', 'Unknown')}\n\n"
                f"They will have access to more resources and will be able to provide you with a comprehensive solution. "
                f"Thank you for your patience!"
    )
    
    return {
        "specialist_info": specialist_info,
        "messages": [escalation_message]
    }

# ============================================================================
# CONDITIONAL ROUTING FUNCTIONS
# ============================================================================

def tools_condition(state: GraphState) -> Literal["tools", "end", "escalate_to_specialist"]:
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("🔄 LLM wants to use tools")
        return "tools"
    else:
        # Check if we have tool results and their quality
        kb_results = state.get("knowledge_base_results", [])
        web_results = state.get("web_search_raw_results", [])
        
        # Calculate scores from tool results
        kb_scores = [r.get('score', 0.0) for r in kb_results if isinstance(r, dict) and 'score' in r]
        web_scores = [r.get('score', 0.0) for r in web_results if isinstance(r, dict) and 'score' in r]
        
        max_kb_score = max(kb_scores) if kb_scores else 0.0
        max_web_score = max(web_scores) if web_scores else 0.0
        
        print(f"📊 Tool result quality: KB={max_kb_score:.1%}, Web={max_web_score:.1%}")
        
        has_good_kb = max_kb_score >= 0.25  # 25% threshold for KB
        has_good_web = max_web_score >= 0.5  # 50% threshold for Web
        has_any_tools_used = len(kb_results) > 0 or len(web_results) > 0
        has_sufficient_info = has_good_kb or has_good_web
        
        if has_any_tools_used and not has_sufficient_info:
            print("🔄 Tool results have low similarity scores - escalating")
            return "escalate_to_specialist"
        else:
            print("🔄 LLM provided final answer - ending conversation")
            return "end"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

builder = StateGraph(GraphState)

# Add nodes
builder.add_node("get_customer_id", get_customer_id)
builder.add_node("get_customer_question", get_customer_question)
builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)
builder.add_node("escalate_to_specialist", escalate_to_specialist)

# Simple linear flow with conditional edges
builder.set_entry_point("get_customer_id")
builder.add_edge("get_customer_id", "get_customer_question")
builder.add_edge("get_customer_question", "llm")

# ReAct-style routing from LLM
builder.add_conditional_edges("llm", tools_condition, {
    "tools": "tools",
    "end": END,
    "escalate_to_specialist": "escalate_to_specialist"
})

# After tools, LLM generates final answer (no loop)
builder.add_edge("tools", "llm")

# Escalation path
builder.add_edge("escalate_to_specialist", END)

# Compile graph for langgraph dev (no checkpointer)
graph = builder.compile()

# Compile graph for interactive demo (with checkpointer for interrupts)
from langgraph.checkpoint.memory import InMemorySaver
interactive_memory = InMemorySaver()
interactive_graph = builder.compile(checkpointer=interactive_memory)

print("✅ Graph compiled successfully")


# ============================================================================
# REAL-WORLD DEMO FUNCTIONS
# ============================================================================

def run_customer_support_demo():
    """Run the real-world customer support demo."""
    print("\n🎭 REAL-WORLD CUSTOMER SUPPORT DEMO")
    print("=" * 60)
    print("This demo simulates a real customer support interaction.")
    print("The agent will ask for your customer ID and question,")
    print("then search for information and decide whether to help")
    print("directly or escalate to a specialist.")
    print("=" * 60)
    
    # Create unique thread for this session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"🆔 Session ID: {thread_id}")
    
    # Step 1: Start the conversation
    print("\n🚀 Starting customer support session...")
    
    result = interactive_graph.invoke({
            "customer_question": "",
            "customer_id": "",
            "customer_context": "",
            "knowledge_base_results": [],
            "web_search_raw_results": [],
            "specialist_info": {},
            "messages": []
        }, config=config)
    
    # Check if interrupt occurred
    if "__interrupt__" in result and result["__interrupt__"]:
        interrupt_info = result["__interrupt__"][0]
        print("✅ First interrupt triggered (Customer ID request)")
        print(f"🤖 Agent: {interrupt_info.value}")
        
        # Step 2: Provide customer ID
        customer_id = input("\n👤 Enter your customer ID (try CUST001, CUST002, etc.): ").strip()
        if not customer_id:
            customer_id = "CUST001"  # Default for demo
        
        result2 = interactive_graph.invoke(Command(resume=customer_id), config=config)
        
        # Check for second interrupt
        if "__interrupt__" in result2 and result2["__interrupt__"]:
            interrupt_info2 = result2["__interrupt__"][0]
            print("✅ Second interrupt triggered (Question request)")
            print(f"🤖 Agent: {interrupt_info2.value}")
            
            # Step 3: Provide question
            print("\n❓ What can we help you with today?")
            print("Try examples like:")
            print("  • 'How do I reset my password?'")
            print("  • 'I'm having API connection issues'")
            print("  • 'I need help with billing'")
            print("  • 'What are your service hours?'")
            
            question = input("\n👤 Your question: ").strip()
            if not question:
                question = "How do I reset my password?"  # Default for demo
            
            print(f"\n🔄 Processing your question: '{question}'")
            print("This may take a moment as we search our knowledge base and web...")
            
            # Step 4: Process the complete workflow
            # Add recursion limit to prevent infinite loops
            config["recursion_limit"] = 10
            final_result = interactive_graph.invoke(Command(resume=question), config=config)
            
            print("\n✅ Customer support session completed!")
            print("=" * 60)
            
            # Display results
            specialist_info = final_result.get("specialist_info", {})
            if specialist_info:
                print("🚨 **ESCALATED TO SPECIALIST**")
                print(f"👤 Specialist: {specialist_info.get('specialist', 'Unknown')}")
                print(f"⏱️ Response time: {specialist_info.get('response_time', 'Unknown')}")
            else:
                print("✅ **RESOLVED DIRECTLY**")
                print("Your question was answered using our knowledge base and web search.")
            
            # Show final message
            messages = final_result.get("messages", [])
            if messages:
                final_message = messages[-1]
                if hasattr(final_message, 'content'):
                    print(f"\n💬 **Final Response:**")
                    print(final_message.content)
            
            # Show search scores
            kb_results = final_result.get("knowledge_base_results", [])
            web_results = final_result.get("web_search_raw_results", [])
            kb_scores = [r.get('score', 0.0) for r in kb_results if isinstance(r, dict) and 'score' in r]
            web_scores = [r.get('score', 0.0) for r in web_results if isinstance(r, dict) and 'score' in r]
            max_kb_score = max(kb_scores) if kb_scores else 0.0
            max_web_score = max(web_scores) if web_scores else 0.0
            print(f"\n📊 **Search Quality Scores:**")
            print(f"   📚 Knowledge Base: {max_kb_score:.1%}")
            print(f"   🌐 Web Search: {max_web_score:.1%}")
            
            return final_result
        else:
            print("❌ Expected second interrupt for question")
            return result2
    else:
        print("❌ No interrupts occurred - check the implementation")
        return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🤖 INTERACTIVE CUSTOMER SUPPORT AGENT")
    print("=" * 50)
    run_customer_support_demo()
    print("\n🎉 Demo completed! Thank you for using our customer support system.")
