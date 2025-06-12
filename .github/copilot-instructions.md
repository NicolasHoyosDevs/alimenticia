# Personalized Nutritional Assistance AI Agent - Project Instructions

## 1. Introduction

This project aims to develop an AI-powered agent for personalized nutritional assistance. Users will interact with a Progressive Web Application (PWA) to receive dietary recommendations based on visual analysis of their meals and their pre-defined nutritional profiles. The backend will leverage Google Gemini models orchestrated by LangGraph.

## 2. Core Technologies

*   **Programming Language:** Python 3.x
*   **AI Models:**
    *   **Gemini 1.5 Flash (or similar, e.g., "Gemini 2.5 Flash" as per proposal):** For image processing (food detection and segmentation).
    *   **Gemini 1.5 Pro (or similar, e.g., "Gemini 2.5 Pro" as per proposal):** For nutritional reasoning and generating personalized recommendations.
    *   **Gemini (e.g., "Gemini 2.0 flash lite" as per proposal, or a suitable chat-optimized model):** For user interaction and Q&A.
*   **Orchestration:** LangGraph
*   **Data Validation & State Management:** Pydantic
*   **Frontend:** Progressive Web Application (PWA) - details to be defined.
*   **Database:** PostgreSQL - For user profiles, meal logs, and chat history.

## 3. Project Setup

1.  **Create a Python Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  **Install Dependencies:**
    Create a `requirements.txt` file:
    ```txt
    langchain
    langchain-google-genai
    langgraph
    pydantic
    google-generativeai
    # Add other dependencies like Flask/FastAPI for PWA backend, DB connectors, etc.
    ```
    Install them:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up Google Gemini API Key:**
    *   Obtain an API key from Google AI Studio.
    *   Set it as an environment variable:
        ```bash
        export GOOGLE_API_KEY="YOUR_API_KEY" # On Windows: set GOOGLE_API_KEY="YOUR_API_KEY"
        ```

## 3.1. Database Schema (PostgreSQL)

The following DDL outlines the structure for the PostgreSQL database.
You will need to define the ENUM types (`sex_enum`, `activity_level_enum`, `meal_type_enum`, `message_sender_enum`) separately in PostgreSQL. For example:
`CREATE TYPE sex_enum AS ENUM ('male', 'female');`
`CREATE TYPE activity_level_enum AS ENUM ('sedentary', 'light', 'moderate', 'active', 'very_active')`
`CREATE TYPE meal_type_enum AS ENUM ('breakfast', 'lunch', 'dinner', 'snack', 'other');`
`CREATE TYPE message_sender_enum AS ENUM ('human', 'ai', 'system');`


```sql
CREATE TABLE user_profiles (
    user_id VARCHAR(255) PRIMARY KEY, -- Could be a UUID or your authentication system's ID

    -- Basic Personal Information
    age INTEGER,
    sex sex_enum,
    height_cm NUMERIC(5,2), -- Ex: 175.50 cm
    current_weight_kg NUMERIC(5,2), -- Ex: 70.50 kg
    activity_level activity_level_enum,

    -- Health and Nutrition Goals
    primary_goal TEXT, -- Ex: "weight_loss", "muscle_gain"
    target_weight_kg NUMERIC(5,2) NULL,
    weight_change_rate_kg_week NUMERIC(4,2) NULL, -- Ex: -0.50 kg/week
    secondary_goals TEXT[] NULL, -- Array of texts, ex: ARRAY['increase fiber intake', 'reduce sugar']

    -- Dietary Restrictions and Preferences
    dietary_restrictions TEXT[] NULL, -- Ex: ARRAY['veganism', 'gluten-free']
    allergies TEXT[] NULL, -- Ex: ARRAY['peanuts', 'shellfish']
    intolerances TEXT[] NULL, -- Ex: ARRAY['lactose']
    disliked_foods TEXT[] NULL,

    -- Relevant Medical Conditions (handle with care due to privacy)
    medical_conditions TEXT[] NULL, -- Ex: ARRAY['type 2 diabetes', 'hypertension']

    -- Nutritional Targets (can be calculated or adjusted)
    target_calories_kcal INTEGER NULL,
    target_protein_g INTEGER NULL,
    target_carbs_g INTEGER NULL,
    target_fat_g INTEGER NULL,
    target_fiber_g INTEGER NULL
);

-- meal_logs table removed as per simplification

CREATE TABLE chat_messages (
    message_id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL, -- Unique identifier for a conversation session
    user_id VARCHAR(255) NOT NULL REFERENCES user_profiles(user_id) ON DELETE CASCADE,
    sender_type message_sender_enum NOT NULL, -- 'human', 'ai', or 'system'
    content TEXT NOT NULL,
    -- Add timestamp for message ordering
    sent_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

## 4. LangGraph Agent Workflow & State

The system will be architected as a multi-agent graph using LangGraph.

### 4.1. Pydantic State Definition

We will use Pydantic to define the shared state object that flows through the graph.

```python
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class DetectedFoodItem(BaseModel):
    label: str = Field(description="Name of the detected food item.")
    bounding_box: List[float] = Field(description="Coordinates of the bounding box [x_min, y_min, x_max, y_max] relative to image size (0.0-1.0) or pixels.")
    # The mask can be complex. For simplicity, we might start with just bounding boxes.
    # The Colab (https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb)
    # shows how Gemini can provide segmentation. This field would store that data.
    mask_data: Optional[Any] = Field(None, description="Segmentation mask data, e.g., RLE, polygon, or raw mask.")
    confidence: Optional[float] = Field(None, description="Detection confidence score.")

class UserProfile(BaseModel):
    user_id: str
    age: Optional[int] = None
    sex: Optional[str] = None # Corresponds to sex_enum from DB
    height_cm: Optional[float] = None
    current_weight_kg: Optional[float] = None
    activity_level: Optional[str] = None # Corresponds to activity_level_enum from DB
    primary_goal: Optional[str] = None
    target_weight_kg: Optional[float] = None
    weight_change_rate_kg_week: Optional[float] = None
    secondary_goals: Optional[List[str]] = Field(default_factory=list)
    dietary_restrictions: Optional[List[str]] = Field(default_factory=list)
    allergies: Optional[List[str]] = Field(default_factory=list)
    intolerances: Optional[List[str]] = Field(default_factory=list)
    disliked_foods: Optional[List[str]] = Field(default_factory=list)
    medical_conditions: Optional[List[str]] = Field(default_factory=list)
    target_calories_kcal: Optional[int] = None
    target_protein_g: Optional[int] = None
    target_carbs_g: Optional[int] = None
    target_fat_g: Optional[int] = None
    target_fiber_g: Optional[int] = None

class AgentState(BaseModel):
    user_id: str
    session_id: Optional[str] = None # Unique identifier for a conversation session
    input_image_bytes: Optional[bytes] = None # Transient for the current session, NOT directly saved to DB
    user_profile: Optional[UserProfile] = None # Fetched from user_profiles table
    detected_foods: Optional[List[DetectedFoodItem]] = Field(default_factory=list) # Transient for the current session, NOT directly saved to DB
    # Store the raw response from Gemini Flash for debugging or richer data
    raw_gemini_image_analysis_response: Optional[Any] = None # Transient for the current session, NOT directly saved to DB
    nutritional_recommendation: Optional[str] = None # Transient for the current session. If part of chat, it's saved via chat_history.
    # For chat history with the Q&A agent, to be related to chat_messages table
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description='e.g., [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]')
    formatted_final_response: Optional[str] = None
    error_message: Optional[str] = None
    # Flags to track progress or conditional routing
    image_analyzed: bool = False
    profile_retrieved: bool = False
    # current_meal_id: Optional[int] = None # Removed as meal_logs table is removed

```

### 4.2. Agent Nodes (Conceptual)

1.  **`WelcomeAndCaptureNode`:**
    *   **Input:** User ID, image from PWA.
    *   **Action:** Initializes the state with `user_id` and `input_image_bytes`.
    *   **Output:** Updates `user_id`, `input_image_bytes` in state.

2.  **`UserProfileRetrievalNode`:**
    *   **Input:** `user_id` from state.
    *   **Action:** Fetches user's nutritional profile from the PostgreSQL database (based on `user_profiles` table) and populates the `UserProfile` model in the state.
    *   **Output:** Updates `user_profile`, `profile_retrieved` in state.

3.  **`ImageAnalysisNode` (Utilizing Gemini 1.5 Flash / "Gemini 2.5 Flash" as per proposal):**
    *   **Input:** `input_image_bytes` from state.
    *   **Action:**
        1.  Prepares the image and text prompts for Gemini.
        2.  Sends prompts to Gemini to detect food items, their bounding boxes, and (optionally) segmentation masks.
            *   **Reference Colab Notebook:** [Spatial Understanding with Gemini](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb#scrollTo=WQJTJ8wdGOKx)
            *   Example prompts (adapt as needed):
                *   "Identify all distinct food items in this image. For each item, provide its name and a bounding box."
                *   "Provide segmentation masks for the identified food items."
        3.  Parses Gemini's response to populate `detected_foods` with `DetectedFoodItem` objects and stores the `raw_gemini_image_analysis_response`. This data is transient in the `AgentState` for the current interaction.
    *   **Output:** Updates `detected_foods`, `raw_gemini_image_analysis_response`, `image_analyzed` in state.

4.  **`NutritionalReasoningNode` (Utilizing Gemini 1.5 Pro / "Gemini 2.5 Pro" as per proposal):**
    *   **Input:** `detected_foods`, `user_profile` from state.
    *   **Action:**
        1.  Constructs a detailed prompt for Gemini, including the list of detected foods, their quantities (inferred from bounding boxes/masks if possible, or qualitative), and the user's complete nutritional profile (goals, restrictions, etc. from the `UserProfile` object).
        2.  Queries Gemini Pro to analyze the meal's composition against the user's goals and generate personalized advice. This recommendation becomes the `nutritional_recommendation` in the state.
    *   **Output:** Updates `nutritional_recommendation` in state.

5.  **`PresentationAndQANode` (Utilizing "Gemini 2.0 flash lite" / chat model as per proposal):**
    *   **Input:** `nutritional_recommendation`, `chat_history` from state.
    *   **Action:**
        1.  Formats the `nutritional_recommendation` for user presentation.
        2.  Handles follow-up questions from the user by interacting with the chat-optimized Gemini model, using the `chat_history` (which can be persisted to/retrieved from the `chat_messages` table) and the context of the recommendation.
    *   **Output:** Updates `formatted_final_response`, `chat_history` in state.

6.  **`DataPersistenceNode`:**
    *   **Input:** `AgentState` (specifically `user_id`, `session_id`, `chat_history`).
    *   **Action:**
        1.  Saves messages from `chat_history` to the `chat_messages` table. Each message is associated with the `user_id` and `session_id`. This includes user's textual inputs and AI's textual responses.
        2.  Image bytes (`input_image_bytes`) and detailed JSON analysis (`raw_gemini_image_analysis_response`, `detected_foods`) are NOT saved to the database.
    *   **Output:** (Potentially status flags if needed, but primarily performs side effects).

### 4.3. LangGraph Edges

*   Define the flow:
    1. `WelcomeAndCaptureNode`
    2. Parallel execution: `UserProfileRetrievalNode` and `ImageAnalysisNode`.
    3. `NutritionalReasoningNode` (after both profile and image analysis are complete).
    4. `PresentationAndQANode` (after nutritional reasoning).
    5. `DataPersistenceNode` (after presentation/Q&A, to log the chat messages for the current session).
*   The `DataPersistenceNode` focuses on chat history. Other data like image analysis results or recommendations are transient within the `AgentState` for the current interaction unless explicitly logged as part of a chat message content (e.g., the AI's textual recommendation).
*   Conditional edges might be needed for error handling or if certain data isn't available.

## 5. Image Processing with Gemini (Key Insights from Colab)

The [Spatial Understanding Colab](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb) is crucial for the `ImageAnalysisNode`.

*   **Multi-modal Prompts:** Gemini can process images and text together. You provide the image and then ask questions about it in text.
*   **Requesting Bounding Boxes:** You can specifically ask Gemini to identify objects and return their bounding boxes. The Colab shows the typical output format (often a list of dictionaries with labels and coordinates).
*   **Requesting Segmentation Masks:** Similarly, you can request segmentation masks. The format and handling of these masks (e.g., as image data, polygons) will need to be determined based on Gemini's output and how you intend to use/display them.
*   **Iterative Prompting:** You might need to refine prompts to get the desired level of detail or accuracy for food items.

## 6. General Guidance & Best Practices

*   **Modularity:** Keep each LangGraph node focused on a single responsibility.
*   **Error Handling:** Implement robust error handling within each node and potentially define error paths in the graph.
*   **Prompt Engineering:** Carefully craft the prompts for each Gemini model to elicit the most accurate and relevant responses.
*   **Configuration:** Manage API keys and other configurations securely (e.g., via environment variables).
*   **Simplicity:** Start with a core functional flow and iterate. For instance, initial versions might focus on bounding boxes before implementing complex segmentation mask processing.
*   **Refer to Gemini Documentation:**
    *   **General Gemini API Usage & Examples:** The [Google Gemini Starter Applets](https://github.com/google-gemini/starter-applets?tab=readme-ov-file) repository can provide useful patterns.
    *   Official Google documentation for the Gemini API and Python SDK.

## 7. Steps (High-Level)

1.  Implement the Pydantic state models.
2.  Develop each LangGraph node as a Python function.
3.  Set up the LangGraph graph with nodes and edges.
4.  Integrate Gemini API calls within the respective nodes, paying close attention to the image analysis part using the Colab notebook as a guide.
5.  Develop a simple PWA interface for user interaction and image upload.
6.  Set up a basic database for user profiles.
7.  Test and iterate.