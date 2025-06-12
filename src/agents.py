# This file will contain the LangGraph agent node implementations.
# Each node will be a Python function that takes the AgentState as input
# and returns a dictionary with updates to the state.

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Assuming state.py is in src
from src.state import AgentState, UserProfile, DetectedFoodItem

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()

# Configure the Gemini API key
# Ensure GOOGLE_API_KEY is set in your .env file or environment variables
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")
genai.configure(api_key=API_KEY)


IMAGE_ANALYSIS_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
NUTRITIONAL_REASONING_MODEL_NAME = "gemini-2.5-pro-preview-05-06"
QA_CHAT_MODEL_NAME = "gemini-2.0-flash-lite"

image_analysis_model = genai.GenerativeModel(IMAGE_ANALYSIS_MODEL_NAME)
nutritional_reasoning_model = genai.GenerativeModel(
    NUTRITIONAL_REASONING_MODEL_NAME)
qa_chat_model = genai.GenerativeModel(QA_CHAT_MODEL_NAME)

# --- Agent Node Implementations ---


def welcome_and_capture_node(state: AgentState) -> dict:
    """Initializes the state with user ID and image (simulated for now)."""
    print("---NODE: Welcome and Capture---")
    # In a real PWA, user_id and image_bytes would come from the request.
    # For now, we might simulate or expect them to be in the initial state invocation.
    if not state.user_id or not state.input_image_bytes:
        # This is a simple way to handle missing initial data for now.
        # A more robust solution would involve how the graph is invoked.
        print("Warning: user_id or input_image_bytes not found in initial state for welcome_and_capture_node.")
        # Potentially set a default or raise an error if critical
        # For this example, we assume they are passed in during graph invocation.

    # No direct state modification here if data is passed in on invocation.
    # This node primarily acts as an entry point and data validator/logger if needed.
    print(
        f"User ID: {state.user_id}, Image Bytes Received: {bool(state.input_image_bytes)}")
    return {}


def user_profile_retrieval_node(state: AgentState) -> dict:
    """Fetches user profile from DB (simulated)."""
    print("---NODE: User Profile Retrieval---")
    user_id = state.user_id
    # TODO: Implement actual database call to PostgreSQL
    print(f"Fetching profile for user_id: {user_id}")
    # Simulated profile data
    profile_data = {
        "user_id": user_id,
        "age": 30,
        "sex": "male",
        "height_cm": 175.0,
        "current_weight_kg": 70.0,
        "activity_level": "moderately_active",
        "primary_goal": "weight_loss",
        "dietary_restrictions": ["lactose_intolerant"],
        "allergies": [],
        "target_calories_kcal": 2000
    }
    user_profile = UserProfile(**profile_data)
    print(f"Profile retrieved: {user_profile}")
    return {"user_profile": user_profile, "profile_retrieved": True}


def image_analysis_node(state: AgentState) -> dict:
    """Analyzes the image using Gemini to detect food items."""
    print("---NODE: Image Analysis---")
    image_bytes = state.input_image_bytes
    if not image_bytes:
        print("Error: No image bytes found for analysis.")
        return {"error_message": "No image provided for analysis.", "image_analyzed": False}

    try:
        print(
            f"Sending image to Gemini ({IMAGE_ANALYSIS_MODEL_NAME}) for analysis...")
        # Prepare the image part for the Gemini API
        # IMPORTANT: The actual structure for `ImagePart` or how you pass image bytes
        # might differ slightly based on the exact `google-generativeai` library version and its API.
        # Refer to the official Gemini Python SDK documentation for the correct way to pass image data.
        # This is a common way:
        image_part = {
            "mime_type": "image/jpeg",  # Or image/png, etc.
            "data": image_bytes
        }

        # Example prompt based on INSTRUCTIONS.md and Colab notebook
        # You can make this more sophisticated
        prompt = """
        Analyze the provided image of a meal.
        Identify all distinct food items in this image.
        For each item, provide its name (label) and a bounding box as a list of four normalized coordinates [ymin, xmin, ymax, xmax].
        Optionally, if you can determine a confidence score for each detection, please include it.
        Return the response as a JSON list of objects, where each object has 'label', 'bounding_box', and optionally 'confidence'.
        Example: [{'label': 'apple', 'bounding_box': [
            0.1, 0.1, 0.3, 0.3], 'confidence': 0.9}]
        """

        # The `generate_content` call for multimodal input (image + text)
        response = image_analysis_model.generate_content([prompt, image_part])

        print(f"Gemini response received for image analysis.")
        # print(f"Raw response text: {response.text}") # For debugging

        # Attempt to parse the response.text which should be JSON
        # This parsing needs to be robust.
        import json
        try:
            # Gemini Pro often returns markdown with JSON, so strip it.
            cleaned_response_text = response.text.strip().removeprefix(
                "```json").removeprefix("```").removesuffix("```")
            detected_items_data = json.loads(cleaned_response_text)

            detected_foods_list = []
            for item_data in detected_items_data:
                # Validate required fields before creating DetectedFoodItem
                if "label" in item_data and "bounding_box" in item_data:
                    detected_foods_list.append(DetectedFoodItem(**item_data))
                else:
                    print(
                        f"Warning: Skipping item due to missing fields: {item_data}")

            print(f"Detected foods: {detected_foods_list}")
            return {
                "detected_foods": detected_foods_list,
                "raw_gemini_image_analysis_response": response.text,  # Store raw for logging
                "image_analyzed": True,
                "error_message": None
            }
        except json.JSONDecodeError as e:
            print(f"Error: Could not parse JSON from Gemini response: {e}")
            print(f"Problematic response text: {response.text}")
            return {"error_message": f"Failed to parse image analysis response: {response.text}", "image_analyzed": False}
        except Exception as e:
            print(f"Error processing Gemini response: {e}")
            return {"error_message": f"Error processing image analysis: {str(e)}", "image_analyzed": False}

    except Exception as e:
        print(f"Error during image analysis with Gemini: {e}")
        # Consider more specific error handling for API errors vs. other issues
        return {"error_message": str(e), "image_analyzed": False}


def nutritional_reasoning_node(state: AgentState) -> dict:
    """Generates nutritional advice using Gemini based on detected food and user profile."""
    print("---NODE: Nutritional Reasoning---")
    detected_foods = state.detected_foods
    user_profile = state.user_profile

    if not state.image_analyzed or not detected_foods:
        return {"error_message": "Image analysis not complete or no food detected."}
    if not state.profile_retrieved or not user_profile:
        return {"error_message": "User profile not retrieved."}

    try:
        # Construct a detailed prompt for Gemini Pro
        food_list_str = ", ".join([food.label for food in detected_foods])
        prompt = f"""
        As a nutritional assistant, analyze the following meal and provide personalized advice.
        User Profile:
        - Age: {user_profile.age}
        - Sex: {user_profile.sex}
        - Height: {user_profile.height_cm} cm
        - Current Weight: {user_profile.current_weight_kg} kg
        - Activity Level: {user_profile.activity_level}
        - Primary Goal: {user_profile.primary_goal}
        - Dietary Restrictions: {', '.join(user_profile.dietary_restrictions) if user_profile.dietary_restrictions else 'None'}
        - Allergies: {', '.join(user_profile.allergies) if user_profile.allergies else 'None'}
        - Target Calories (if available): {user_profile.target_calories_kcal or 'Not specified'} kcal

        Detected Foods in the Meal: {food_list_str}
        (Note: Assume typical portion sizes unless otherwise inferable from a general image context. Detailed portion size/weight is not available from the image analysis.)

        Based on the user's profile and the detected foods, provide a concise and actionable nutritional recommendation.
        Consider the user's primary goal. For example, if the goal is weight loss, suggest adjustments for that.
        If the meal seems well-aligned, acknowledge that. If there are concerns, point them out and suggest improvements.
        Keep the recommendation to 2-4 sentences.
        """
        print(
            f"Sending prompt to Gemini ({NUTRITIONAL_REASONING_MODEL_NAME}) for nutritional reasoning...")
        response = nutritional_reasoning_model.generate_content(prompt)
        recommendation = response.text
        print(f"Nutritional recommendation: {recommendation}")
        return {"nutritional_recommendation": recommendation, "error_message": None}

    except Exception as e:
        print(f"Error during nutritional reasoning with Gemini: {e}")
        return {"error_message": str(e)}


def presentation_and_qa_node(state: AgentState) -> dict:
    """Formats the recommendation and handles Q&A (simulated)."""
    print("---NODE: Presentation and Q&A---")
    recommendation = state.nutritional_recommendation
    # Access directly, default_factory handles initialization
    chat_history = state.chat_history

    if not recommendation:
        # If there's an error from a previous step, reflect that.
        error_msg = state.error_message or "No recommendation available."
        formatted_response = f"Sorry, I couldn't generate a recommendation at this time. Error: {error_msg}"
        chat_history.append({"role": "model", "content": formatted_response})
        return {"formatted_final_response": formatted_response, "chat_history": chat_history}

    # Simple formatting for now
    formatted_response = f"""Here's your nutritional insight:
{recommendation}

What else can I help you with regarding this meal or your nutritional goals?"""

    # Add current agent response to chat history
    chat_history.append({"role": "model", "content": formatted_response})

    # TODO: Implement actual Q&A loop with the QA_CHAT_MODEL_NAME
    # This would involve taking user input, appending to chat_history,
    # calling qa_chat_model.generate_content(chat_history_formatted_for_gemini),
    # and then updating formatted_final_response and chat_history again.
    # For now, we just present the initial recommendation.

    print(f"Formatted response: {formatted_response}")
    return {"formatted_final_response": formatted_response, "chat_history": chat_history, "error_message": None}


def data_persistence_node(state: AgentState) -> dict:
    """Saves chat history to DB (simulated)."""
    print("---NODE: Data Persistence---")
    user_id = state.user_id
    session_id = state.session_id  # Get session_id from state

    if not user_id or not session_id:
        print("Error: user_id or session_id missing for data persistence.")
        return {"error_message": "User ID or Session ID missing for persistence."}

    # Log chat messages
    chat_history = state.chat_history  # Access directly
    if chat_history:
        print(
            f"Persisting chat messages for user {user_id}, session {session_id}...")
        # TODO: Implement actual DB insert into 'chat_messages'
        # Ensure you have a way to avoid logging duplicate messages if this node can be re-run for the same state.
        # One common strategy is to only log new messages since the last persistence point for this session.
        for message in chat_history:  # This assumes chat_history contains all messages for the session
            # db_insert_chat_message({
            #     "user_id": user_id,
            #     "session_id": session_id,
            #     "sender_type": map_role_to_sender_type(message["role"]), # e.g., 'model' -> 'ai'
            #     "content": message["content"]
            # })
            pass  # Replace with actual DB call
        print(
            f"{len(chat_history)} messages processed for DB (simulated for session {session_id}).")
    else:
        print("No chat history to persist.")

    # No direct state changes by default, primarily side effects
    return {"error_message": None}

# Placeholder for DB utility functions (to be implemented in a separate db_utils.py or similar)
# def db_insert_chat_message(data):
#     pass

# def map_role_to_sender_type(role):
#    if role == "model":
#        return "ai"
#    elif role == "user":
#        return "human"
#    return "system" # Or handle other roles as needed
