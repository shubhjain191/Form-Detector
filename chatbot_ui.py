import pandas as pd
import cohere
import os
import time
import logging
from dotenv import load_dotenv
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()
cohere_api_key = os.environ.get("COHERE_API_KEY")

# Add debug logging for API key
if not cohere_api_key:
    st.error("❌ Cohere API key not found! Please check your .env file.")

try:
    co = cohere.Client(cohere_api_key)
except Exception as e:
    st.error(f"❌ Error initializing Cohere client: {str(e)}")

# --- Dataset Loading ---
def load_exercise_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        return df
    except FileNotFoundError:
        st.error("📁 Dataset file not found. Please check the file path.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"📁 Error loading dataset: {str(e)}")
        return pd.DataFrame()

# Load the dataset
exercise_data = load_exercise_data('megaGymDataset.csv')

# --- Need Follow-Up Logic ---
def need_follow_up(query):
    # Simple follow-up detection based on keywords
    if "more" in query.lower() or "details" in query.lower():
        return "It seems like you want more details. Could you specify what exactly you'd like to know?"
    return None

# --- Process User Query ---
def process_query(query, exercise_data, user_preferences):
    try:
        # Debug logging using logging module
        logging.debug(f"Processing query: {query}")
        logging.debug(f"User preferences: {user_preferences}")

        # First, check if it's a follow-up question needed
        follow_up = need_follow_up(query)
        if follow_up:
            return {"type": "follow_up", "message": follow_up}

        if "describe" in query.lower():
            exercise_name = extract_exercise_name(query)
            description = describe_exercise(exercise_name, exercise_data)
            return {"type": "description", "message": description if description else f"Sorry, I couldn't find details about '{exercise_name}'."}
        else:
            response = generate_response(query, user_preferences)
            return {"type": "general", "message": response}
    except Exception as e:
        error_msg = f"❌ Error processing query: {str(e)}"
        st.error(error_msg)
        return {"type": "error", "message": error_msg}

def generate_response(query, user_preferences):
    try:
        prompt = craft_fitness_prompt(query, user_preferences)

        # Debug logging
        logging.debug(f"Generated prompt: {prompt}")

        response = co.generate(
            model='command-nightly',
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            stop_sequences=["--"]
        )

        # Debug logging
        logging.debug(f"Raw Cohere response: {response}")

        if response and hasattr(response, 'generations') and response.generations:
            return response.generations[0].text.strip()
        else:
            return "I apologize, but I couldn't generate a response. Please try rephrasing your question."

    except Exception as e:
        error_msg = f"⚠️ Error generating response: {str(e)}"
        st.error(error_msg)
        return error_msg

def craft_fitness_prompt(query, user_preferences):
    prompt = (
        f"You are a knowledgeable and friendly fitness expert chatbot. "
        f"Provide a detailed and helpful response to the following question, "
        f"taking into account the user's preferences:\n\n"
        f"User Profile:\n"
        f"- Goal: {user_preferences['goal']}\n"
        f"- Experience Level: {user_preferences['experience']}\n"
        f"- Available Time: {user_preferences['available_time']}\n"
        f"- Workout Frequency: {user_preferences['workout_frequency']}\n"
        f"- Equipment Access: {', '.join(user_preferences['equipment_access'])}\n"
        f"- Restrictions: {user_preferences['restrictions']}\n\n"
        f"User Question: {query}\n\n"
        f"Please provide a detailed, helpful, and encouraging response with relevant fitness advice. "
        f"Include specific recommendations when appropriate."
    )
    return prompt

# Dummy functions for extract_exercise_name and describe_exercise
# These would need to be properly implemented based on your dataset and logic.
def extract_exercise_name(query):
    # This is a placeholder. You'd implement NLP to extract exercise names.
    # For example, a simple keyword search:
    common_exercises = ["squats", "push-ups", "bicep curls", "shoulder press", "deadlifts", "bench press"]
    for exercise in common_exercises:
        if exercise in query.lower():
            return exercise.replace('-', ' ').title() # Format nicely
    return None

def describe_exercise(exercise_name, exercise_data):
    if exercise_name and not exercise_data.empty:
        # Assuming 'Title' is the column for exercise names and 'Desc' for descriptions
        exercise_row = exercise_data[exercise_data['Title'].str.contains(exercise_name, case=False, na=False)]
        if not exercise_row.empty:
            return exercise_row.iloc[0]['Desc']
    return None

# --- Streamlit UI ---
def chat_ui():
    # --- CSS for Chatbot UI ---
    st.markdown("""
        <style>
        /* Inherit global styles from main.py if loaded together, or define here for standalone */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Open+Sans:wght@400;600&display=swap');

        :root {
            --primary-color: #007bff; /* Professional Blue */
            --secondary-color: #28a745; /* Success Green */
            --accent-color: #ffc107; /* Warning Yellow */
            --text-color: #333333; /* Darker text for readability */
            --text-color-light: #6c757d; /* Muted text */
            --bg-color-main: #f8f9fa; /* Light grey background */
            --bg-color-sidebar: #ffffff;
            --bg-color-card: #ffffff;
            --border-color: #e9ecef; /* Light border */
            --border-radius: 10px; /* Consistent border radius */
            --shadow-sm: 0 .125rem .25rem rgba(0,0,0,.075);
            --shadow-md: 0 .5rem 1rem rgba(0,0,0,.1);
            --shadow-lg: 0 1rem 3rem rgba(0,0,0,.175);
            --font-family-heading: 'Montserrat', sans-serif;
            --font-family-body: 'Open Sans', sans-serif;
            --gradient-primary: linear-gradient(90deg, #007bff, #0056b3); /* Deeper blue gradient */
            --gradient-secondary: linear-gradient(90deg, #28a745, #1e7e34);
        }

        body { font-family: var(--font-family-body); color: var(--text-color); }

        .chatbot-title {
            font-family: var(--font-family-heading);
            font-size: 2.8rem;
            font-weight: 700;
            color: var(--primary-color);
            text-align: center;
            margin-bottom: 2.5rem;
            position: relative;
            padding-bottom: 1rem;
        }
        .chatbot-title::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 80px;
            height: 4px;
            background: var(--gradient-primary);
            border-radius: 3px;
        }

        .stAlert { /* Consistent info box styling */
            border-radius: var(--border-radius) !important;
            border: none !important;
            border-left: 6px solid !important;
            box-shadow: var(--shadow-md) !important;
            padding: 1.5rem 2rem !important;
            margin-bottom: 2rem;
            font-family: var(--font-family-body);
        }
        .stAlert[kind="info"] { border-left-color: var(--primary-color) !important; background-color: #e6f2ff !important; color: #0056b3 !important;}

        .stExpander { /* Consistent expander styling */
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            box-shadow: var(--shadow-sm);
            margin-bottom: 2rem;
        }
        .stExpander summary {
            font-family: var(--font-family-heading);
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--primary-color);
            padding: 0.8rem 1rem;
            border-radius: var(--border-radius);
            background-color: rgba(0, 123, 255, 0.05);
            border: none; /* Summary itself doesn't need border, parent div has it */
            transition: background-color 0.2s ease;
        }
        .stExpander summary:hover {
            background-color: rgba(0, 123, 255, 0.1);
        }
        .stExpander details {
            padding: 1.5rem;
            font-family: var(--font-family-body);
            color: var(--text-color-light);
        }

        /* Chat Message Styling */
        .chat-message {
            padding: 1rem 1.2rem;
            border-radius: var(--border-radius);
            margin-bottom: 1rem;
            line-height: 1.6;
            font-family: var(--font-family-body);
            box-shadow: var(--shadow-sm);
        }
        .chat-message.user {
            background-color: #e0f2f7; /* Light blue for user */
            color: var(--text-color);
            text-align: right;
            border-bottom-right-radius: 0;
            margin-left: 20%; /* Keep messages on right */
        }
        .chat-message.bot {
            background-color: var(--bg-color-card);
            color: var(--text-color);
            text-align: left;
            border: 1px solid var(--border-color);
            border-bottom-left-radius: 0;
            margin-right: 20%; /* Keep messages on left */
        }
        .chat-message strong {
            font-weight: 700;
            color: var(--primary-color); /* Highlight speaker */
        }

        /* Input and Button Styling */
        div[data-testid="stTextInput"] > div > div > input {
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            padding: 0.8rem 1rem;
            font-size: 1rem;
            box-shadow: var(--shadow-sm);
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="stTextInput"] > div > div > input:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
            outline: none;
        }

        div[data-testid="stButton"] > button {
            background: var(--gradient-primary);
            color: white;
            padding: 0.9rem 2rem;
            border-radius: 50px;
            border: none;
            font-weight: 600;
            font-size: 1.05rem;
            transition: transform 0.2s ease, box-shadow 0.3s ease, background-position 0.4s ease;
            box-shadow: var(--shadow-md);
            width: auto;
            background-size: 200% auto;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
            background-position: right center;
        }

        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] button {
            font-family: var(--font-family-heading);
            font-weight: 600;
            font-size: 1.1rem;
            color: var(--text-color-light);
            background-color: var(--bg-color-main);
            border-radius: var(--border-radius) var(--border-radius) 0 0;
            border: 1px solid var(--border-color);
            border-bottom: none;
            padding: 1rem 1.5rem;
            transition: all 0.2s ease;
            margin-right: 0.5rem;
        }
        .stTabs [data-baseweb="tab-list"] button:hover {
            color: var(--primary-color);
            background-color: var(--bg-color-card);
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: var(--primary-color);
            background-color: var(--bg-color-card);
            border-top: 3px solid var(--primary-color); /* Active tab indicator */
            font-weight: 700;
            box-shadow: var(--shadow-sm);
        }
        .stTabs [data-baseweb="tab-panel"] {
            border: 1px solid var(--border-color);
            border-top: none;
            border-radius: 0 0 var(--border-radius) var(--border-radius);
            padding: 2rem;
            background-color: var(--bg-color-card);
            box-shadow: var(--shadow-md);
        }


        /* User Preferences Styling (within sidebar if used as standalone) */
        .st-emotion-cache-nahz7x label { /* Label for selectbox, radio, multiselect */
            font-family: var(--font-family-heading);
            font-weight: 600;
            color: var(--text-color);
            margin-bottom: 0.5rem;
            display: block;
            font-size: 1rem;
        }
        .st-emotion-cache-nahz7x div[data-testid="stSelectbox"],
        .st-emotion-cache-nahz7x div[data-testid="stMultiSelect"],
        .st-emotion-cache-nahz7x div[data-testid="stRadio"] {
            margin-bottom: 1.5rem;
        }

        .st-emotion-cache-nahz7x div[data-testid="stSelectbox"] div[data-baseweb="select"] {
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
        }
        .st-emotion-cache-nahz7x div[data-testid="stSelectbox"] div[data-baseweb="select"]:hover {
            border-color: var(--primary-color);
        }

        /* Radio buttons from main.py's sidebar style */
        .st-emotion-cache-nahz7x div[data-testid="stRadio"] > div {
            flex-direction: column;
            gap: 0.6rem;
        }
        .st-emotion-cache-nahz7x div[data-testid="stRadio"] label[data-baseweb="radio"] {
            background-color: var(--bg-color-main) !important;
            border-radius: 8px;
            padding: 0.7rem 1.2rem;
            border: 1px solid var(--border-color);
            transition: all 0.2s ease-in-out;
            width: 100%;
            box-shadow: var(--shadow-sm);
        }
        .st-emotion-cache-nahz7x div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
            border-color: var(--primary-color);
            background-color: rgba(0, 123, 255, 0.08) !important;
            box-shadow: var(--shadow-md);
            color: var(--primary-color);
        }
        .st-emotion-cache-nahz7x div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
            border-color: var(--primary-color);
            background-color: rgba(0, 123, 255, 0.05) !important;
            box-shadow: var(--shadow-sm);
        }
        .st-emotion-cache-nahz7x div[data-testid="stRadio"] label[data-baseweb="radio"] span:first-child { display: none; }
        .st-emotion-cache-nahz7x div[data-testid="stRadio"] label[data-baseweb="radio"] span:last-child::before {
            content: ''; display: inline-block; width: 18px; height: 18px;
            border: 2px solid var(--text-color-light); border-radius: 50%;
            margin-right: 10px; vertical-align: middle; transition: all 0.2s ease-in-out;
        }
        .st-emotion-cache-nahz7x div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) span:last-child::before {
            background-color: var(--primary-color); border-color: var(--primary-color);
            box-shadow: 0 0 0 4px rgba(0, 123, 255, 0.2);
        }

        .st-emotion-cache-nahz7x div[data-testid="stSlider"] div[data-baseweb="slider"] {
            height: 8px;
            background-color: var(--border-color);
            border-radius: 4px;
        }
        .st-emotion-cache-nahz7x div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
            background-color: var(--primary-color);
            border: 2px solid #fff;
            box-shadow: var(--shadow-sm);
        }

        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="chatbot-title">🏋️‍♀️ Your AI Fitness Coach</h1>', unsafe_allow_html=True)

    # Add an introduction
    st.info("""
    **Empower your fitness journey with personalized advice from our AI coach.**
    Ask about:
    * **Exercise Techniques:** Master proper form for any movement.
    * **Workout Strategies:** Design effective routines and optimize your training.
    * **Injury Prevention:** Learn how to stay safe and recover smarter.
    * **Nutrition & Wellness:** Get insights into healthy eating and holistic fitness.
    """, icon="💬")

    # Gather user preferences in a sidebar for better organization
    st.sidebar.header("🎯 Set Your Fitness Profile")
    user_preferences = gather_user_preferences_sidebar() # New function for sidebar preferences

    # Show current profile summary in main area
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
    with st.expander("👤 **YOUR CURRENT PROFILE**"):
        st.markdown(f"""
            <p style='font-family: var(--font-family-body); font-size: 1rem; color: var(--text-color);'>
                <strong>Goal</strong>: {user_preferences['goal']} 🎯<br>
                <strong>Experience</strong>: {user_preferences['experience']} 💪<br>
                <strong>Workout Time</strong>: {user_preferences['available_time']} ⏰<br>
                <strong>Frequency</strong>: {user_preferences['workout_frequency']} days/week 📅<br>
                <strong>Equipment</strong>: {', '.join(user_preferences['equipment_access'])} 🏋️‍♂️<br>
                <strong>Restrictions</strong>: {user_preferences['restrictions'] if user_preferences['restrictions'] else 'None'} 🚫
            </p>
        """, unsafe_allow_html=True)

    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)

    # User query input with placeholder
    user_input = st.text_input(
        "Ask me about workouts, exercises, or fitness tips!",
        placeholder="E.g., 'Suggest a full-body workout for beginners' or 'What are the benefits of squats?'",
        key="user_query_input"
    )

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Create tabs for current conversation and history
    tab_conversation, tab_history = st.tabs(["💬 CURRENT CONVERSATION", "📜 CHAT HISTORY"])

    with tab_conversation:
        # Display current conversation messages
        for chat_message in st.session_state.chat_history:
            if chat_message["user"] == user_input and chat_message["bot"] is not None:
                # This is the last interaction, display it now
                if chat_message["bot"]["type"] == "error":
                    st.markdown(f"<div class='chat-message user'><strong>You</strong>: {chat_message['user']}</div>", unsafe_allow_html=True)
                    st.error(chat_message["bot"]["message"])
                elif chat_message["bot"]["type"] == "follow_up":
                    st.markdown(f"<div class='chat-message user'><strong>You</strong>: {chat_message['user']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='chat-message bot'><strong>Bot</strong>: {chat_message['bot']['message']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-message user'><strong>You</strong>: {chat_message['user']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='chat-message bot'><strong>Bot</strong>: {chat_message['bot']['message']}</div>", unsafe_allow_html=True)
            else:
                 # Display past messages
                if chat_message["bot"]["type"] == "error":
                    st.markdown(f"<div class='chat-message user'><strong>You</strong>: {chat_message['user']}</div>", unsafe_allow_html=True)
                    st.error(chat_message["bot"]["message"])
                elif chat_message["bot"]["type"] == "follow_up":
                    st.markdown(f"<div class='chat-message user'><strong>You</strong>: {chat_message['user']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='chat-message bot'><strong>Bot</strong>: {chat_message['bot']['message']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-message user'><strong>You</strong>: {chat_message['user']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='chat-message bot'><strong>Bot</strong>: {chat_message['bot']['message']}</div>", unsafe_allow_html=True)


        if st.button("SEND MESSAGE 🚀"):
            if not user_input:
                st.warning("⚠️ Please enter a question or query.")
            else:
                with st.spinner("Thinking... 🤔"):
                    response = process_query(user_input, exercise_data, user_preferences)

                    timestamp = int(time.time() * 1000)
                    st.session_state.chat_history.append({
                        "user": user_input,
                        "bot": response,
                        "timestamp": timestamp
                    })
                st.experimental_rerun() # Rerun to display new message
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)

        with st.expander("💡 **TIPS FOR BETTER RESPONSES**"):
            st.markdown("""
            <div class="tips-box">
                <ul>
                    <li><strong>Be Specific:</strong> Clearly state the exercise or fitness topic you're interested in.</li>
                    <li><strong>Mention Equipment:</strong> Indicate what equipment you have access to for tailored workout suggestions.</li>
                    <li><strong>Include Your Level:</strong> Provide your fitness level (beginner, intermediate, advanced) for personalized routines.</li>
                    <li><strong>Ask for Alternatives:</strong> If an exercise doesn't suit you, ask for suitable variations.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab_history:
        st.write("### Complete Chat History")
        if st.session_state.chat_history:
            # Display history in reverse chronological order
            for chat in reversed(st.session_state.chat_history):
                # Optionally format timestamp:
                # from datetime import datetime
                # dt_object = datetime.fromtimestamp(chat["timestamp"] / 1000)
                # st.markdown(f"<small style='color: #888;'>{dt_object.strftime('%Y-%m-%d %H:%M:%S')}</small>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-message user'><strong>You</strong>: {chat['user']}</div>", unsafe_allow_html=True)
                if chat["bot"]["type"] == "error":
                    st.error(chat["bot"]["message"])
                elif chat["bot"]["type"] == "follow_up":
                    st.markdown(f"<div class='chat-message bot'><strong>Bot</strong>: {chat['bot']['message']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-message bot'><strong>Bot</strong>: {chat['bot']['message']}</div>", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.info("No conversation history yet. Start chatting in the 'Current Conversation' tab!")


# --- User Preferences (Moved to Sidebar) ---
def gather_user_preferences_sidebar():
    goal = st.sidebar.selectbox("What's your main fitness goal?",
                        ["Weight Loss", "Build Muscle", "Endurance", "General Fitness"],
                        key="pref_goal")
    experience = st.sidebar.radio("What's your experience level?",
                          ["Beginner", "Intermediate", "Advanced"],
                          key="pref_experience")
    available_time = st.sidebar.selectbox("How much time can you dedicate to each workout?",
                                  ["< 30 minutes", "30-45 minutes", "45-60 minutes", "60+ minutes"],
                                  key="pref_time")
    workout_frequency = st.sidebar.slider("How many days a week do you plan to workout?",
                                  min_value=1, max_value=7, value=3,
                                  key="pref_frequency")
    equipment_access = st.sidebar.multiselect("What equipment do you have access to?",
                                      ["Dumbbells", "Barbell", "Kettlebells", "Resistance Bands", "Bodyweight", "Machines (Gym)"],
                                      key="pref_equipment")
    restrictions = st.sidebar.text_input("Do you have any injury or limitation? (Optional)",
                                         key="pref_restrictions")

    return {
        "goal": goal,
        "experience": experience,
        "available_time": available_time,
        "workout_frequency": workout_frequency,
        "equipment_access": equipment_access,
        "restrictions": restrictions
    }


# Entry point for Streamlit
if __name__ == "__main__":
    st.set_page_config(
        page_title="AI Fitness Coach", # Consistent page title
        page_icon="🏋️‍♂️", # Consistent icon
        layout="wide",
        initial_sidebar_state="expanded"
    )
    chat_ui()