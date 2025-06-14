from dotenv import load_dotenv
import traceback  # Added for more detailed error logging in nodes
import uuid  # For session_id generation if needed, though typically done by API layer


# Assuming schema.py is in src (relative import)
from ..schemas.schema import AgentState, UserProfile, DetectedFoodItem


# Import services
from ..services.gemini_service import (
    analyze_image_with_gemini,
    get_nutritional_advice,
    get_chat_response
)
from src.services.database_service import get_connection
# TODO: Import database_service when its functions are used


# Load environment variables (like GOOGLE_API_KEY, which gemini_service will use)
load_dotenv()


# --- Agent Node Implementations ---


# welcome_and_capture_node has been removed as per updated instructions.
# Initial AgentState (including session_id, user_id, input_image_bytes, meal_context_description)
# should be created by the API layer (e.g., FastAPI route) before invoking the LangGraph workflow.




def user_profile_retrieval_node(state: AgentState) -> dict:
    """Obtiene el perfil del usuario desde la base de datos usando el user_id del estado."""
    print("---NODE: User Profile Retrieval---")
    user_id = state.user_id
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT user_id, "password", created_at, updated_at, age, sex, height_cm, current_weight_kg, activity_level,
                   primary_goal, target_weight_kg, weight_change_rate_kg_week, secondary_goals, dietary_restrictions,
                   allergies, intolerances, medical_conditions, target_calories_kcal, target_protein_g, target_carbs_g,
                   target_fat_g, target_fiber_g
            FROM public.user_profiles
            WHERE user_id = %s
        """, (user_id,))
        result = cur.fetchone()
        cur.close()
        conn.close()

        if not result:
            return {"user_profile": None, "profile_retrieved": False, "error_message": "Usuario no encontrado"}

        keys = [
            "user_id", "password", "created_at", "updated_at", "age", "sex", "height_cm", "current_weight_kg", "activity_level",
            "primary_goal", "target_weight_kg", "weight_change_rate_kg_week", "secondary_goals", "dietary_restrictions",
            "allergies", "intolerances", "medical_conditions", "target_calories_kcal", "target_protein_g", "target_carbs_g",
            "target_fat_g", "target_fiber_g"
        ]
        profile_data = dict(zip(keys, result))

        # Convierte los campos tipo array a listas de Python si es necesario
        for field in ["secondary_goals", "dietary_restrictions", "allergies", "intolerances", "medical_conditions"]:
            if profile_data.get(field) is not None and not isinstance(profile_data[field], list):
                profile_data[field] = list(profile_data[field])

        # Filtra solo los campos que espera el modelo UserProfile
        user_profile_fields = [
            "user_id", "age", "sex", "height_cm", "current_weight_kg", "activity_level",
            "primary_goal", "target_weight_kg", "weight_change_rate_kg_week", "secondary_goals",
            "dietary_restrictions", "allergies", "intolerances", "medical_conditions", "target_calories_kcal",
            "target_protein_g", "target_carbs_g", "target_fat_g", "target_fiber_g"
        ]
        user_profile_data = {k: profile_data[k] for k in user_profile_fields if k in profile_data}

        user_profile = UserProfile(**user_profile_data)
        return {"user_profile": user_profile.model_dump(), "profile_retrieved": True, "error_message": None}
    except Exception as e:
        print(f"Error al obtener el perfil del usuario: {e}")
        traceback.print_exc()
        return {"user_profile": None, "profile_retrieved": False, "error_message": f"Error al obtener el perfil: {str(e)}"}




def image_analysis_node(state: AgentState) -> dict:
    """Analyzes the image using the gemini_service to detect food items.
    Returns detected_foods as a list of dictionaries.
    """
    print("---NODE: Image Analysis---")
    image_bytes = state.input_image_bytes


    if not image_bytes:
        print("Error: No image bytes found for analysis.")
        return {"error_message": "No image provided for analysis.", "image_analyzed": False, "detected_foods": []}


    try:
        print(f"Calling gemini_service.analyze_image_with_gemini...")
        # gemini_service.analyze_image_with_gemini returns List[DetectedFoodItem]
        detected_foods_models, raw_gemini_image_analysis_response = analyze_image_with_gemini(
            image_bytes=image_bytes)


        if detected_foods_models is None:
            print("Error: Image analysis service returned None.")
            return {"error_message": "Image analysis service failed.", "image_analyzed": False, "detected_foods": []}


        print(f"Detected foods (models): {detected_foods_models}")
        # Convert list of models to list of dicts for LangGraph state
        detected_foods_dicts = [item.model_dump()
                                for item in detected_foods_models]

        

        return {
            "detected_foods": detected_foods_dicts,
            "raw_gemini_image_analysis_response": raw_gemini_image_analysis_response,
            "image_analyzed": True,
            "error_message": None
        }
    except Exception as e:
        print(f"Error during image analysis node: {e}")
        traceback.print_exc()
        return {"error_message": str(e), "image_analyzed": False, "detected_foods": []}




def nutritional_reasoning_node(state: AgentState) -> dict:
    """Generates nutritional advice using gemini_service based on detected food and user profile."""
    print("---NODE: Nutritional Reasoning---")


    # Reconstruct models from dictionaries in state
    detected_foods_models = []
    if state.detected_foods:
        try:
            detected_foods_models = [
                item if isinstance(item, DetectedFoodItem) else DetectedFoodItem(**item)
                for item in state.detected_foods
            ]
        except Exception as e:
            error_msg = f"Error reconstructing DetectedFoodItem models: {e}"
            print(f"Error: {error_msg}")
            traceback.print_exc()
            return {"error_message": error_msg, "nutritional_recommendation": None, "recommendation_generated": False}


    user_profile_model = None
    if state.user_profile:
        try:
            if isinstance(state.user_profile, UserProfile):
                user_profile_model = state.user_profile
            else:
                user_profile_model = UserProfile(**state.user_profile)
        except Exception as e:
            error_msg = f"Error reconstructing UserProfile model: {e}"
            print(f"Error: {error_msg}")
            traceback.print_exc()
            return {"error_message": error_msg, "nutritional_recommendation": None, "recommendation_generated": False}


    # meal_context = state.meal_context_description


    if not state.image_analyzed or not detected_foods_models:  # Check models list
        error_msg = "Image analysis not complete or no food detected (or failed reconstruction) for nutritional reasoning."
        print(f"Error: {error_msg}")
        return {"error_message": error_msg, "nutritional_recommendation": None, "recommendation_generated": False}
    if not state.profile_retrieved or not user_profile_model:  # Check model
        error_msg = "User profile not retrieved (or failed reconstruction) for nutritional reasoning."
        print(f"Error: {error_msg}")
        return {"error_message": error_msg, "nutritional_recommendation": None, "recommendation_generated": False}


    try:
        print(f"Calling gemini_service.get_nutritional_advice with models...")
        recommendation = get_nutritional_advice(
            detected_foods=detected_foods_models,
            user_profile=user_profile_model,
            # meal_context=meal_context
        )


        if recommendation is None:
            print("Error: Nutritional reasoning service returned None.")
            return {"error_message": "Nutritional reasoning service failed.", "nutritional_recommendation": None, "recommendation_generated": False}


        print(f"Nutritional recommendation: {recommendation}")
        return {"nutritional_recommendation": recommendation, "error_message": None, "recommendation_generated": True}


    except Exception as e:
        print(f"Error during nutritional reasoning node: {e}")
        traceback.print_exc()
        return {"error_message": str(e), "nutritional_recommendation": None, "recommendation_generated": False}




def presentation_node(state: AgentState) -> dict:
    """Formatea la recomendación final para el usuario."""
    print("---NODE: Presentation---")
    recommendation = state.nutritional_recommendation
    formatted_response = None
    if recommendation:
        formatted_response = f"Recomendación nutricional personalizada:\n{recommendation}"
    else:
        formatted_response = "No se pudo generar una recomendación nutricional."
    # Actualiza el historial de chat solo con la respuesta final (opcional)
    new_chat_history = state.chat_history.copy()
    new_chat_history.append({"role": "model", "content": formatted_response})
    return {
        "formatted_final_response": formatted_response,
        "chat_history": new_chat_history
    }




def data_persistence_node(state: AgentState) -> dict:
    """Saves chat history to DB (simulated)."""
    print("---NODE: Data Persistence---")
    user_id = state.user_id
    session_id = state.session_id
    chat_history_to_persist = state.chat_history


    if not user_id or not session_id:
        print("Error: user_id or session_id missing for data persistence.")
        return {"error_message": "User ID or Session ID missing for persistence."}


    if chat_history_to_persist:
        print(
            f"Persisting chat messages for user {user_id}, session {session_id}...")
        # TODO: Implement actual DB insert into 'chat_messages' using database_service
        # For each message in chat_history_to_persist:
        #   db_service.save_message(user_id, session_id, message['role'], message['content'])
        # Ensure to map 'role' to 'sender_type' enum for the database.


        # Example of mapping and simulated persistence:
        for message in chat_history_to_persist:
            sender_type = map_role_to_sender_type(message["role"])
            # print(f"  Simulating save: User: {user_id}, Session: {session_id}, Sender: {sender_type}, Content: '{message['content'][:50]}...'")


        print(
            f"{len(chat_history_to_persist)} total messages in current history for session {session_id} considered for DB (simulated).")
    else:
        print("No chat history to persist.")


    return {"error_message": None}




def map_role_to_sender_type(role: str) -> str:
    if role == "model":
        return "ai"
    elif role == "user":
        return "human"
    return "system"
