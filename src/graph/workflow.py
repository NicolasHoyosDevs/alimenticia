from langgraph.graph import StateGraph, END
from src.schemas.schema import AgentState
from src.agents.agents import (
    user_profile_retrieval_node,
    image_analysis_node,
    nutritional_reasoning_node,
    presentation_node,
    data_persistence_node
)
import uuid  # For generating session_id in the test example
import traceback  # For printing tracebacks


# --- LangGraph Workflow Definition ---


# Initialize the StateGraph with the AgentState Pydantic model
workflow = StateGraph(AgentState)


# Add nodes to the graph
# welcome_and_capture_node is removed.
workflow.add_node("user_profile_retrieval", user_profile_retrieval_node)
workflow.add_node("image_analysis", image_analysis_node)
workflow.add_node("nutritional_reasoning", nutritional_reasoning_node)
workflow.add_node("presentation", presentation_node)
workflow.add_node("data_persistence", data_persistence_node)


# Define the entry point of the graph
# user_profile_retrieval_node is now the entry point.
workflow.set_entry_point("user_profile_retrieval")


# Define edges for a sequential flow:
# UserProfileRetrieval -> ImageAnalysis -> NutritionalReasoning -> Presentation -> DataPersistence
workflow.add_edge("user_profile_retrieval", "image_analysis")
workflow.add_edge("image_analysis", "nutritional_reasoning")
workflow.add_edge("nutritional_reasoning", "presentation")
workflow.add_edge("presentation", "data_persistence")


# Define the end point of the graph
workflow.add_edge("data_persistence", END)


# Compile the graph into a runnable application
app = workflow.compile()


# --- Example of how to run the graph (for testing) ---
# This would typically be part of your main application logic,
# perhaps triggered by an API endpoint.


if __name__ == "__main__":
    print("Compiling and running LangGraph agent...")


    sample_user_id = "10"
    sample_session_id = "ses_10_20250613173534_d10f37c1"
    sample_meal_context = "My breakfast"


    try:
        with open(r"C:\Users\MSI\Pictures\comida1.jpg", "rb") as f:
            sample_image_bytes = f.read()
    except FileNotFoundError:
        print("Warning: Test image not found. Using placeholder bytes for image_bytes.")
        sample_image_bytes = b"simulated_image_data_bytes_placeholder"


    initial_state_data = {
        "user_id": str(sample_user_id),
        "session_id": sample_session_id,
        "input_image_bytes": sample_image_bytes,
        "meal_context_description": sample_meal_context,
        "user_profile": None,
        "detected_foods": [],
        "raw_gemini_image_analysis_response": None,
        "nutritional_recommendation": None,
        "chat_history": [],
        "formatted_final_response": None,
        "error_message": None,
        "profile_retrieved": False,
        "image_analyzed": False,
        "recommendation_generated": False
    }
    initial_state_obj = AgentState(**initial_state_data)


    # Prepare a printable version of the initial state
    initial_state_dict_for_print = initial_state_obj.model_dump()
    printable_initial_state = {
        k: (f'{len(v) if isinstance(v, bytes) else len(str(v)) if v is not None else 0} bytes'
            if k == 'input_image_bytes'
            else 'response_placeholder'
            if k == 'raw_gemini_image_analysis_response'
            else v)
        for k, v in initial_state_dict_for_print.items()
    }
    print(f"Invoking graph with initial state: {printable_initial_state}")


    # This will hold the full state as a dictionary, evolving with each step
    current_full_state_dict = initial_state_obj.model_dump()
    # This will hold the AgentState object corresponding to the latest full state
    final_state_result_object = initial_state_obj


    try:
        # Pass the initial state dictionary to stream
        for step_output_event in app.stream(current_full_state_dict):
            # step_output_event is a dictionary like: {'node_name': {'field1': 'value1', ...}}
            node_name = next(iter(step_output_event))
            # These are the updates from the node
            node_output_dict = step_output_event[node_name]


            print(f"--- Output from node '{node_name}' ---")
            # Print the partial update for clarity (can be made prettier if needed)
            print(node_output_dict)


            if node_output_dict is not None and isinstance(node_output_dict, dict):
                # Merge the node's output (updates) into our copy of the full state dictionary
                current_full_state_dict.update(node_output_dict)


                # Create a new AgentState instance from the *updated full state dictionary*
                try:
                    final_state_result_object = AgentState(
                        **current_full_state_dict)
                except Exception as e_pydantic:
                    print(
                        f"Pydantic validation error when reconstructing state after node {node_name}: {e_pydantic}")
                    print(
                        f"State dictionary that caused error: {current_full_state_dict}")
                    traceback.print_exc()
                    raise  # Re-raise the error to stop execution and see the problem


                print(
                    f"--- Current Full State after node '{node_name}' (reconstructed) ---")
                # Print the full state (or a summary)
                printable_full_state = {
                    k: (f'{len(v) if isinstance(v, bytes) else len(str(v)) if v is not None else 0} bytes/chars' if k == 'input_image_bytes'
                        else 'response_placeholder' if k == 'raw_gemini_image_analysis_response'
                        else v)
                    for k, v in final_state_result_object.model_dump().items()
                }
                print(printable_full_state)
            else:
                print(
                    f"Warning: Output from node '{node_name}' is None or not a dict. Actual: {type(node_output_dict)}")
                # If a node returns None, state effectively doesn't change from its perspective,
                # final_state_result_object remains as is from the previous valid state.


            print("---------------------------------------")


        # After the loop, final_state_result_object holds the state after the last successful node execution
        if final_state_result_object is not None and isinstance(final_state_result_object, AgentState):
            print("\n---FINAL AGENT OUTPUT---")
            print(
                f"Formatted Response: {final_state_result_object.formatted_final_response}")
            if final_state_result_object.error_message:
                print(f"Error: {final_state_result_object.error_message}")
        elif final_state_result_object is not None:
            print("\n---FINAL AGENT OUTPUT (Unexpected format)---")
            print(f"Final state: {final_state_result_object}")
        else:
            print("Graph execution did not produce a final state (it was None). This likely means the last node or an intermediate node returned None, or the stream ended prematurely.")


    except Exception as e:
        print(f"An error occurred during graph execution: {e}")
        traceback.print_exc()


    print("\nGraph execution finished.")

# from langgraph.graph import StateGraph, END
# from schemas.schema import AgentState
# from src.agents import (
#     welcome_and_capture_node,
#     user_profile_retrieval_node,
#     image_analysis_node,
#     nutritional_reasoning_node,
#     presentation_node,
#     data_persistence_node
# )

# # --- LangGraph Workflow Definition ---

# workflow = StateGraph(AgentState)

# workflow.add_node("welcome_and_capture", welcome_and_capture_node)
# workflow.add_node("user_profile_retrieval", user_profile_retrieval_node)
# workflow.add_node("image_analysis", image_analysis_node)
# workflow.add_node("nutritional_reasoning", nutritional_reasoning_node)
# workflow.add_node("presentation", presentation_node)
# workflow.add_node("data_persistence", data_persistence_node)

# workflow.set_entry_point("welcome_and_capture")

# workflow.add_edge("welcome_and_capture", "user_profile_retrieval")
# workflow.add_edge("welcome_and_capture", "image_analysis")
# workflow.add_edge("user_profile_retrieval", "nutritional_reasoning")
# workflow.add_edge("image_analysis", "nutritional_reasoning")
# workflow.add_edge("nutritional_reasoning", "presentation")
# workflow.add_edge("presentation", "data_persistence")
# workflow.add_edge("data_persistence", END)

# app = workflow.compile()

# if __name__ == "__main__":
#     print("Compiling and running LangGraph agent...")

#     # Simulate initial input
#     # Replace with actual image loading
#     # with open(r"path_to_your_image.jpg", "rb") as f:
#     #     sample_image_bytes = f.read()
#     sample_image_bytes = b"simulated_image_data_bytes"

#     initial_state = {
#         "user_id": "test_user_123",
#         "input_image_bytes": sample_image_bytes,
#         "chat_history": []
#     }
#     print(
#         f"Invoking graph with initial state: {{k: v if k != 'input_image_bytes' else 'image_bytes_present' for k, v in initial_state.items()}}")

#     final_state = None
#     try:
#         for step_output in app.stream(initial_state):
#             node_name = next(iter(step_output))
#             current_state_after_node = step_output[node_name]

#             print(f"--- Current State after node '{node_name}' ---")

#             if current_state_after_node is not None:
#                 state_dict = None
#                 if isinstance(current_state_after_node, AgentState):
#                     state_dict = current_state_after_node.model_dump()
#                 elif isinstance(current_state_after_node, dict):
#                     state_dict = current_state_after_node
#                 else:
#                     print(
#                         f"Warning: State after node '{node_name}' is of unexpected type: {type(current_state_after_node)}")
#                     state_dict = {}

#                 printable_state = {
#                     k: v if k != 'input_image_bytes' else f'{len(v) if isinstance(v, bytes) else "0 or not bytes"} bytes'
#                     for k, v in state_dict.items()
#                 }
#                 print(printable_state)
#             else:
#                 print(
#                     f"Warning: State after node '{node_name}' is None.")

#             print("---------------------------------------")
#             final_state = current_state_after_node

#         if final_state is not None and isinstance(final_state, AgentState):
#             print("\n---FINAL AGENT OUTPUT---")
#             print(
#                 f"Formatted Response: {final_state.formatted_final_response}")
#             if final_state.error_message:
#                 print(f"Error: {final_state.error_message}")
#         elif final_state is not None:
#             print("\n---FINAL AGENT OUTPUT (Unexpected format)---")
#             print(f"Final state: {final_state}")
#         else:
#             print("Graph execution did not produce a final state.")

#     except Exception as e:
#         print(f"An error occurred during graph execution: {e}")
#         import traceback
#         traceback.print_exc()

#     print("\nGraph execution finished.")
