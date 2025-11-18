from google.adk.runners import Runner
from google.genai import types

print("✅ ADK components imported successfully.")

# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
    USER_ID: str = "",
):
    print(f"\n ### Session: {session_name}")
    # Get app name from the Runner
    app_name = runner_instance.app_name
    session_service = runner_instance.session_service

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"assistant > ", event.content.parts[0].text)

        # for debugging
        # rep = await runner_instance.run_debug(user_id=USER_ID, session_id=session.id, user_messages=user_queries)
        # print(rep)
    else:
        print("No queries!")



async def run_session_return(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
    USER_ID: str = "",
):
    print(f"\n ### Session: {session_name}")
    # Get app name from the Runner
    app_name = runner_instance.app_name
    session_service = runner_instance.session_service

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]
        resp = ""
        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(user_id=USER_ID, session_id=session.id, new_message=query):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (event.content.parts[0].text != "None" and event.content.parts[0].text):
                        resp += event.content.parts[0].text 
                        print(f"assistant > ", event.content.parts[0].text)
        return resp
     
    else:
        print("No queries!")


print("✅ Helper functions defined.")