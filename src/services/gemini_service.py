import os
import json
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import google.generativeai as genai
from ..schemas.schema import DetectedFoodItem, UserProfile  # Adjusted import path

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()

# Configure the Gemini API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")
genai.configure(api_key=API_KEY)


IMAGE_ANALYSIS_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
NUTRITIONAL_REASONING_MODEL_NAME = "gemini-2.5-pro-preview-05-06"
QA_CHAT_MODEL_NAME = "gemini-2.0-flash-lite"

# Initialize models
# These will be initialized once when the module is imported.
try:
    image_analysis_model = genai.GenerativeModel(IMAGE_ANALYSIS_MODEL_NAME)
    nutritional_reasoning_model = genai.GenerativeModel(
        NUTRITIONAL_REASONING_MODEL_NAME)
    qa_chat_model = genai.GenerativeModel(QA_CHAT_MODEL_NAME)
    print("Gemini models initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini models: {e}")
    # Depending on the application's needs, you might want to raise the error
    # or handle it in a way that allows the app to run with limited functionality.
    image_analysis_model = None
    nutritional_reasoning_model = None
    qa_chat_model = None


def analyze_image_with_gemini(image_bytes: bytes, custom_prompt: Optional[str] = None) -> List[DetectedFoodItem]:
    """
    Analyzes an image using Gemini to detect food items, their bounding boxes,
    and potentially other information as requested by the prompt.

    Args:
        image_bytes: The image data in bytes.
        custom_prompt: An optional custom prompt to guide the Gemini model.
                       If None, a default prompt for food detection is used.

    Returns:
        A list of DetectedFoodItem objects.
        Returns an empty list if an error occurs or no items are detected.
    """
    if not image_analysis_model:
        print("Error: Image analysis model not initialized.")
        return []
    if not image_bytes:
        print("Error: No image bytes provided for analysis.")
        return []

    try:
        print(
            f"Sending image to Gemini ({IMAGE_ANALYSIS_MODEL_NAME}) for analysis...")

        image_part = {
            # Assuming JPEG, adjust if necessary (e.g., "image/png")
            "mime_type": "image/jpeg",
            "data": image_bytes
        }

        if custom_prompt:
            prompt = custom_prompt
        else:
            # Default prompt based on INSTRUCTIONS.md and Colab for food detection
            prompt = """
            Analyze the provided image of a meal.
            Identify all distinct food items in this image.
            For each item, provide its name (label) and a bounding box as a list of four normalized coordinates [ymin, xmin, ymax, xmax].
            The response MUST be a valid JSON list of objects, where each object has 'label' and 'bounding_box'.
            Example: [{"label": "apple", "bounding_box": [0.1, 0.1, 0.3, 0.3]}, {"label": "banana", "bounding_box": [0.4, 0.4, 0.6, 0.6]}]
            If no food items are clearly identifiable, return an empty list [].
            """

        # The generate_content call for multimodal input (image + text)
        # As per the Colab, the arguments are passed as a list.
        response = image_analysis_model.generate_content([prompt, image_part])
        raw_gemini_image_analysis_response = response.text

        print("Gemini response received for image analysis.")
        # print(f"Raw response text: {response.text}") # For debugging

        # Attempt to parse the response.text which should be JSON
        cleaned_response_text = response.text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.startswith("```"):
            cleaned_response_text = cleaned_response_text[3:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]

        parsed = safe_json_loads(cleaned_response_text)
        if parsed is None:
            return {
                "error_message": "No se pudo analizar la respuesta de Gemini (JSON inválido)",
                "image_analyzed": False,
                "detected_foods": []
            }

        detected_foods_list = []
        if isinstance(parsed, list):  # Ensure the response is a list
            for item_data in parsed:
                if isinstance(item_data, dict) and "label" in item_data and "bounding_box" in item_data:
                    # Optional: Add more validation for bounding_box format if needed
                    detected_foods_list.append(DetectedFoodItem(**item_data))
                else:
                    print(
                        f"Warning: Skipping item due to missing fields or incorrect format: {item_data}")
        else:
            print(
                f"Warning: Expected a list from Gemini, but got: {type(parsed)}")

        print(f"Detected foods: {detected_foods_list}")
        return detected_foods_list, raw_gemini_image_analysis_response

    except json.JSONDecodeError as e:
        print(f"Error: Could not parse JSON from Gemini response: {e}")
        print(
            f"Problematic response text: {response.text if hasattr(response, 'text') else 'No response text available'}")
        return []
    except Exception as e:
        print(f"Error during image analysis with Gemini: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_nutritional_advice(detected_foods: List[DetectedFoodItem], user_profile: UserProfile) -> Optional[str]:
    """
    Generates nutritional advice using Gemini based on detected food and user profile.

    Args:
        detected_foods: A list of DetectedFoodItem objects.
        user_profile: The UserProfile object.

    Returns:
        A string containing the nutritional recommendation, or None if an error occurs.
    """
    if not nutritional_reasoning_model:
        print("Error: Nutritional reasoning model not initialized.")
        return None
    if not detected_foods:
        print("Info: No detected foods to analyze for nutritional advice.")
        return "No food items were detected, so I cannot provide specific nutritional advice for this meal."
    if not user_profile:
        print("Error: User profile not provided for nutritional advice.")
        return None

    try:
        food_list_str = ", ".join([food.label for food in detected_foods])

        # Constructing a more detailed prompt
        # prompt_parts = [
        #     "As a nutritional assistant, analyze the following meal and provide personalized advice.",
        #     "User Profile:",
        #     f"- Age: {user_profile.age or 'Not specified'}",
        #     f"- Sex: {user_profile.sex or 'Not specified'}",
        #     f"- Height: {user_profile.height_cm or 'Not specified'} cm",
        #     f"- Current Weight: {user_profile.current_weight_kg or 'Not specified'} kg",
        #     f"- Activity Level: {user_profile.activity_level or 'Not specified'}",
        #     f"- Primary Goal: {user_profile.primary_goal or 'Not specified'}",
        #     f"- Dietary Restrictions: {', '.join(user_profile.dietary_restrictions) if user_profile.dietary_restrictions else 'None'}",
        #     f"- Allergies: {', '.join(user_profile.allergies) if user_profile.allergies else 'None'}",
        #     f"- Intolerances: {', '.join(user_profile.intolerances) if user_profile.intolerances else 'None'}",
        #     f"- Disliked Foods: {', '.join(user_profile.disliked_foods) if user_profile.disliked_foods else 'None'}",
        #     f"- Medical Conditions: {', '.join(user_profile.medical_conditions) if user_profile.medical_conditions else 'None'}",
        #     f"- Target Calories (kcal): {user_profile.target_calories_kcal or 'Not specified'}",
        #     f"- Target Protein (g): {user_profile.target_protein_g or 'Not specified'}",
        #     f"- Target Carbs (g): {user_profile.target_carbs_g or 'Not specified'}",
        #     f"- Target Fat (g): {user_profile.target_fat_g or 'Not specified'}",
        #     f"- Target Fiber (g): {user_profile.target_fiber_g or 'Not specified'}",
        #     f"\nDetected Foods in the Meal: {food_list_str}",
        #     "\n(Note: Assume typical portion sizes. Detailed portion size/weight is not available from the image analysis.)",
        #     "\nBased on the user's profile and the detected foods, provide a concise and actionable nutritional recommendation.",
        #     "Consider the user's primary goal. For example, if the goal is weight loss, suggest adjustments for that.",
        #     "If the meal seems well-aligned, acknowledge that. If there are concerns, point them out and suggest improvements.",
        #     "Keep the recommendation to 2-4 clear and helpful sentences."
        # ]
        prompt_parts = [
            "You are an advanced AI Nutritional Assistant, acting with the knowledge and reasoning capabilities of an expert human nutritionist. Your goal is to provide highly personalized, insightful, and actionable advice.",
    "A user has submitted an image of their meal, and an initial visual analysis has identified the following food items and their relative detected sizes (proxied by bounding box area; a larger number suggests a larger portion relative to other items in the same image):",
    f"{food_list_str}",
    "\nHere is the user's detailed nutritional profile:",
    f"  - Age: {user_profile.age or 'Not specified'}",
    f"  - Sex: {user_profile.sex or 'Not specified'}",
    f"  - Height: {user_profile.height_cm or 'Not specified'} cm",
    f"  - Current Weight: {user_profile.current_weight_kg or 'Not specified'} kg",
    f"  - Activity Level: {user_profile.activity_level or 'Not specified'}",
    f"  - Primary Goal: {user_profile.primary_goal or 'Not specified'}",
    f"  - Target Weight: {user_profile.target_weight_kg or 'Not specified'} kg",
    f"  - Secondary Goals: {', '.join(user_profile.secondary_goals) if user_profile.secondary_goals else 'None'}",
    f"  - Dietary Restrictions: {', '.join(user_profile.dietary_restrictions) if user_profile.dietary_restrictions else 'None'}",
    f"  - Allergies: {', '.join(user_profile.allergies) if user_profile.allergies else 'None'}",
    f"  - Intolerances: {', '.join(user_profile.intolerances) if user_profile.intolerances else 'None'}",
    f"  - Disliked Foods: {', '.join(user_profile.disliked_foods) if user_profile.disliked_foods else 'None'}",
    f"  - Medical Conditions: {', '.join(user_profile.medical_conditions) if user_profile.medical_conditions else 'None'}",
    f"  - Target Daily Calories (kcal): {user_profile.target_calories_kcal or 'To be determined'}",
    f"  - Target Daily Protein (g): {user_profile.target_protein_g or 'To be determined'}",
    f"  - Target Daily Carbs (g): {user_profile.target_carbs_g or 'To be determined'}",
    f"  - Target Daily Fat (g): {user_profile.target_fat_g or 'To be determined'}",
    f"  - Target Daily Fiber (g): {user_profile.target_fiber_g or 'To be determined'}",

    "\nINSTRUCTIONS FOR ANALYSIS AND RECOMMENDATION:",
    "1.  *Infer Relative Portion Sizes:* Critically consider the 'Detected Size Proxy' for each food item. While not an exact grammage, use it to infer the relative abundance of each component in the meal. For example, a 'pasta' item with a large size proxy likely constitutes a significant portion of the meal's carbohydrates.",
    "2.  *Deep Nutritional Evaluation of Each Food:* For each identified food, access your extensive nutritional knowledge. Consider its typical macronutrient (protein, carbs, fat) and micronutrient (vitamins, minerals) profile, fiber content, glycemic index, and potential benefits or drawbacks (e.g., 'salmon' is high in omega-3s; 'white bread' is a refined carb).",
    "3.  *Holistic Meal Assessment:* Evaluate the meal as a whole. Does it appear balanced? Does it contain a good source of protein, complex carbohydrates, healthy fats, and sufficient fiber? Are there any glaring omissions or excesses?",
    "4.  *Contextualize with User Profile:* This is CRUCIAL. Analyze the meal IN THE CONTEXT of the user's specific profile: their goals (e.g., weight loss, muscle gain, diabetes management), restrictions, allergies, and overall nutritional targets. For example, a meal مناسب for muscle gain might be too calorie-dense for someone aiming for weight loss.",
    "5.  *Identify Alignments and Misalignments:* Clearly state how the meal aligns or misalign with the user's goals and nutritional needs. Be specific. For example: 'The grilled chicken is a good source of lean protein, aligning with your muscle gain goal. However, the large portion of fries may add excessive unhealthy fats and calories, potentially hindering your progress if not accounted for.'",
    "6.  *Provide Actionable, Prioritized Recommendations:* Offer 2-4 concrete, actionable suggestions for improvement if needed. These should be realistic and easy for the user to implement. Examples:",
    "    *   'Consider reducing the portion of pasta by about a third and adding more non-starchy vegetables like spinach or zucchini to increase fiber and nutrients while managing carbohydrate intake for your weight loss goal.'",
    "    *   'To boost protein for your muscle gain objective, you could add another egg or a side of Greek yogurt to this breakfast.'",
    "    *   'This meal looks quite balanced for your current maintenance goal! Ensure you're incorporating a variety of colorful vegetables throughout the day.'",
    "7.  *Tone and Language:* Maintain a supportive, empathetic, and encouraging tone. Avoid judgmental language. Explain your reasoning clearly but concisely.",
    "8.  *Output Format:* Present the analysis and recommendations in a clear, easy-to-read format. Use bullet points or short paragraphs for readability.",
    "\nBegin your response with a brief overall impression of the meal, followed by your detailed analysis and recommendations."

        ]



        prompt = "\\n".join(prompt_parts)

        print(
            f"Sending prompt to Gemini ({NUTRITIONAL_REASONING_MODEL_NAME}) for nutritional reasoning...")
        response = nutritional_reasoning_model.generate_content(prompt)
        recommendation = response.text.strip()
        print(f"Nutritional recommendation received: {recommendation}")
        return recommendation

    except Exception as e:
        print(f"Error during nutritional reasoning with Gemini: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_chat_response(chat_history: List[Dict[str, str]], current_recommendation: Optional[str] = None) -> Optional[str]:
    """
    Handles follow-up questions using a chat-optimized Gemini model.

    Args:
        chat_history: A list of chat messages, e.g., 
                      [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}].
                      The roles should be 'user' and 'model' as expected by Gemini.
        current_recommendation: The latest nutritional recommendation to provide context.

    Returns:
        A string containing the AI's response, or None if an error occurs.
    """
    if not qa_chat_model:
        print("Error: Q&A chat model not initialized.")
        return None
    if not chat_history:
        print("Error: Chat history is empty.")
        return None

    try:
        # The Gemini API expects a list of Content parts.
        # We need to convert our chat history format.
        # The 'model' role is used for the AI's responses.

        # Construct a context string to prepend to the history if needed,
        # or ensure the history itself provides enough context.
        # For this example, we'll assume the chat_history is directly usable
        # if formatted correctly (list of alternating user/model messages).

        # The Colab and documentation show that for `generate_content` with history,
        # the history should be part of the `contents` list.
        # The last message in the history is typically the user's latest query.

        # If a current recommendation is available and not yet in history,
        # it might be good to prepend it as system/model context,
        # but let's keep it simple and assume it's part of the flow leading to the Q&A.

        print(
            f"Sending chat history to Gemini ({QA_CHAT_MODEL_NAME}) for Q&A response...")
        # Ensure roles are 'user' and 'model'
        gemini_formatted_history = []
        for message in chat_history:
            role = message.get("role")
            content = message.get("content")
            if role == "user" or role == "model":  # Gemini uses 'user' and 'model'
                gemini_formatted_history.append(
                    {'role': role, 'parts': [{'text': content}]})
            # else: skip or handle system messages if any

        if not gemini_formatted_history:
            print("Warning: Chat history was empty after formatting for Gemini.")
            return "I don't have any previous conversation to respond to."

        response = qa_chat_model.generate_content(gemini_formatted_history)
        ai_response = response.text.strip()
        print(f"Q&A response received: {ai_response}")
        return ai_response

    except Exception as e:
        print(f"Error during Q&A with Gemini: {e}")
        import traceback
        traceback.print_exc()
        return None


def safe_json_loads(response_text):
    try:
        # Limpia el texto si viene con ```json ... ```
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)
    except Exception as e:
        print(f"Error al parsear JSON: {e}")
        print(f"Texto problemático: {response_text}")
        return None


# Example Usage (for testing this module directly)
if __name__ == '__main__':
    print("--- Testing Gemini Service ---")

    # Test 1: Image Analysis (requires a sample image)
    print("\\n--- Test 1: Image Analysis ---")
    sample_image_path = "test_image.jpg"  # Create a dummy image or use a real one
    try:
        # Create a dummy image file for testing if it doesn't exist
        if not os.path.exists(sample_image_path):
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (100, 100), color='red')
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), "Test", fill=(0, 0, 0))
                img.save(sample_image_path, "JPEG")
                print(f"Created dummy '{sample_image_path}' for testing.")
            except ImportError:
                print(
                    "Pillow not installed, cannot create dummy image. Please create 'test_image.jpg' manually.")
                # As a fallback, skip image analysis if Pillow is not there and no image exists
                if not os.path.exists(sample_image_path):
                    raise FileNotFoundError(
                        "test_image.jpg not found and Pillow not available to create it.")

        with open(sample_image_path, "rb") as f:
            sample_image_bytes = f.read()

        if sample_image_bytes:
            detected_foods = analyze_image_with_gemini(sample_image_bytes)
            if detected_foods:
                print("Detected food items:")
                for food in detected_foods:
                    print(
                        f"- {food.label} (Confidence: {food.confidence or 'N/A'}) at {food.bounding_box}")
            else:
                print("No food items detected or an error occurred.")
        else:
            print("Could not load sample image bytes.")
            detected_foods = []  # Ensure detected_foods is defined for the next test

    except FileNotFoundError:
        print(
            f"Error: Test image '{sample_image_path}' not found. Skipping image analysis test.")
        detected_foods = []  # Initialize for next test
    except Exception as e:
        print(f"An error occurred during image analysis test: {e}")
        detected_foods = []

    # Test 2: Nutritional Reasoning
    print("\\n--- Test 2: Nutritional Reasoning ---")
    # Use detected_foods from Test 1 if available, otherwise mock some
    if not detected_foods:  # If image analysis failed or returned empty
        print("Mocking detected foods for nutritional reasoning test as previous step failed or yielded no results.")
        detected_foods = [
            DetectedFoodItem(label="Cooked Chicken Breast",
                             bounding_box=[0.1, 0.1, 0.5, 0.5]),
            DetectedFoodItem(label="Steamed Broccoli",
                             bounding_box=[0.2, 0.6, 0.4, 0.8])
        ]

    sample_user_profile = UserProfile(
        user_id="test_user_service",
        age=30,
        sex="female",
        height_cm=165.0,
        current_weight_kg=60.0,
        activity_level="light",
        primary_goal="maintain_weight",
        target_calories_kcal=1800
    )
    advice = get_nutritional_advice(detected_foods, sample_user_profile)
    if advice:
        print(f"Nutritional Advice: {advice}")
    else:
        print("Could not get nutritional advice.")

    # Test 3: Q&A Chat
    print("\\n--- Test 3: Q&A Chat ---")
    sample_chat_history = [
        {"role": "user", "content": "What about the vitamins in this meal?"}
    ]
    if advice:  # Add the previous advice as model's turn if available
        sample_chat_history.insert(0, {"role": "model", "content": advice})

    chat_response = get_chat_response(sample_chat_history)
    if chat_response:
        print(f"Chatbot Response: {chat_response}")
    else:
        print("Could not get chat response.")

    print("\\n--- Gemini Service Testing Finished ---")
