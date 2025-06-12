from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

# --- Database Schema Aligned Pydantic Models ---


class DetectedFoodItem(BaseModel):
    label: str = Field(description="Name of the detected food item.")
    bounding_box: List[float] = Field(
        description="Coordinates of the bounding box [x_min, y_min, x_max, y_max] relative to image size (0.0-1.0) or pixels.")
    mask_data: Optional[Any] = Field(
        None, description="Segmentation mask data, e.g., RLE, polygon, or raw mask.")
    confidence: Optional[float] = Field(
        None, description="Detection confidence score.")


class UserProfile(BaseModel):
    user_id: str
    age: Optional[int] = None
    sex: Optional[str] = None  # Corresponds to sex_enum from DB
    height_cm: Optional[float] = None
    current_weight_kg: Optional[float] = None
    # Corresponds to activity_level_enum from DB
    activity_level: Optional[str] = None
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

# --- LangGraph Agent State ---


class AgentState(BaseModel):
    user_id: str
    # Unique identifier for a conversation session
    session_id: Optional[str] = None
    # Current image for analysis in the session
    input_image_bytes: Optional[bytes] = None
    # Fetched from user_profiles table
    user_profile: Optional[UserProfile] = None
    detected_foods: Optional[List[DetectedFoodItem]] = Field(
        default_factory=list)  # Current session's detected foods
    # Current session's raw analysis
    raw_gemini_image_analysis_response: Optional[Any] = None
    # Current session's recommendation
    nutritional_recommendation: Optional[str] = None
    chat_history: List[Dict[str, str]] = Field(
        # To be related to chat_messages table
        default_factory=list, description='e.g., [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]')
    formatted_final_response: Optional[str] = None
    error_message: Optional[str] = None
    image_analyzed: bool = False
    profile_retrieved: bool = False
