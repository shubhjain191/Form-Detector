# --- START OF FILE main.py ---

import streamlit as st
import time
import ExerciseAiTrainer as exercise # Assuming ExerciseAiTrainer handles its own imports
from chatbot_ui import chat_ui
from streamlit_option_menu import option_menu # Using option-menu for a cleaner sidebar switch potentially

# --- Helper function to switch view ---
def set_view(view_name):
    st.session_state.current_view = view_name

def main():
    # Page configuration - sets up the browser tab appearance
    st.set_page_config(
        page_title='FormFit AI - Elite Performance Training', # More professional title
        page_icon='🏋️', # Professional weightlifting icon
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # --- Initialize Session State ---
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'landing' # Start with the landing page view

    # --- Advanced CSS for a Premium, Modern UI/UX ---
    st.markdown("""
        <style>
        /* === Font Import === */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Open+Sans:wght@400;600&display=swap');

        /* === CSS Variables for Theming === */
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

        /* === Global Styles === */
        body {
            font-family: var(--font-family-body) !important;
            color: var(--text-color);
            background-color: var(--bg-color-main);
            line-height: 1.6;
        }
        .stApp > header {
            background-color: transparent;
            height: 0px; /* Effectively hide header */
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 1000;
        }
        .stApp {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        /* === Landing Page Specific Styles === */
        .landing-container {
            padding: 0rem 3rem 5rem 3rem; /* More generous padding */
        }

        .landing-hero {
            text-align: center;
            padding: 5rem 2rem 5rem 2rem; /* Larger hero section */
            margin: -1rem -3rem 4rem -3rem; /* Extend slightly */
            background: linear-gradient(135deg, rgba(0, 123, 255, 0.08), rgba(40, 167, 69, 0.05));
            border-bottom: 1px solid var(--border-color);
            border-radius: var(--border-radius); /* Rounded hero corners */
        }
        .main-header {
            font-family: var(--font-family-heading);
            font-size: 4.5rem; /* Even larger */
            font-weight: 800;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1.5rem;
            line-height: 1.1;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.05); /* Subtle text shadow */
        }
        .subtitle {
            font-family: var(--font-family-body);
            font-size: 1.6rem; /* Larger */
            color: var(--text-color-light);
            margin-bottom: 3.5rem;
            font-weight: 400;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.5;
        }

        /* === Section Headers === */
        .section-header {
            font-family: var(--font-family-heading);
            font-size: 2.5rem; /* Larger section headers */
            font-weight: 700;
            color: var(--primary-color); /* Highlight with primary color */
            margin-bottom: 3.5rem;
            text-align: center;
            position: relative;
            padding-bottom: 1rem;
        }
        .section-header::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100px; /* Wider and more prominent underline */
            height: 5px; /* Thicker underline */
            background: var(--gradient-primary);
            border-radius: 3px;
        }

        /* === Mode Selection Cards === */
        .mode-selection-container {
            display: flex;
            gap: 2.5rem; /* More space between cards */
            justify-content: center;
            margin-bottom: 5rem;
        }
        .mode-card {
            background: var(--bg-color-card);
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            padding: 3rem 2.5rem; /* More padding */
            text-align: center;
            width: 100%;
            box-shadow: var(--shadow-md);
            transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
            overflow: hidden;
        }
        .mode-card::before { /* Decorative top border */
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 5px;
            background: var(--gradient-primary);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .mode-card:hover {
            transform: translateY(-10px);
            box-shadow: var(--shadow-lg);
            border-color: var(--primary-color);
        }
        .mode-card:hover::before {
            opacity: 1;
        }
        .mode-icon {
            font-size: 4rem; /* Larger icon */
            margin-bottom: 1.8rem;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1;
        }
        .mode-title {
            font-family: var(--font-family-heading);
            font-size: 1.8rem; /* Larger title */
            font-weight: 700;
            margin-bottom: 1rem;
            color: var(--text-color);
        }
        .mode-description {
            font-family: var(--font-family-body);
            color: var(--text-color-light);
            font-size: 1rem;
            line-height: 1.7;
            margin-bottom: 2.5rem;
            flex-grow: 1;
        }
        .mode-card div[data-testid="stButton"] > button {
             margin-top: auto;
             width: 90%; /* Wider button within card */
             background: var(--gradient-primary); /* Ensure consistent button style */
        }

        /* === Feature Cards === */
        .feature-card {
            padding: 2.5rem 2rem; /* More padding */
            border-radius: var(--border-radius);
            background: var(--bg-color-card);
            box-shadow: var(--shadow-sm); /* Lighter shadow initially */
            margin-bottom: 2rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            border: 1px solid var(--border-color);
        }
        .feature-card:hover {
            transform: translateY(-8px);
            box-shadow: var(--shadow-md); /* More pronounced shadow on hover */
            border-color: var(--primary-color);
        }
        .feature-icon {
            font-size: 3.5rem; /* Larger icon */
            margin-bottom: 1.5rem;
            background: var(--gradient-primary); /* Primary gradient for features */
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
        }
        .feature-title {
            font-family: var(--font-family-heading);
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            color: var(--text-color);
        }
        .feature-description {
            font-family: var(--font-family-body);
            color: var(--text-color-light);
            font-size: 0.95rem;
            line-height: 1.6;
            flex-grow: 1;
        }

        /* === Buttons Styling === */
        div[data-testid="stButton"] > button {
            background: var(--gradient-primary);
            color: white;
            padding: 1rem 2.2rem; /* More padding */
            border-radius: 50px;
            border: none;
            font-weight: 600;
            font-size: 1.1rem; /* Slightly larger font */
            transition: transform 0.2s ease, box-shadow 0.3s ease, background-position 0.4s ease;
            box-shadow: var(--shadow-md);
            width: auto;
            background-size: 200% auto;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.7rem;
            cursor: pointer;
            text-transform: uppercase; /* Professional look */
            letter-spacing: 0.05em; /* Spacing for uppercase */
        }
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-5px); /* More pronounced lift */
            box-shadow: var(--shadow-lg);
            background-position: right center;
        }
        div[data-testid="stButton"] > button:active {
            transform: translateY(-2px);
            box-shadow: var(--shadow-sm);
        }
        .stButton button[kind="primary"] {
            background: var(--gradient-secondary); /* Use secondary color for primary action */
            padding: 1.1rem 2.8rem; /* Even larger */
            font-size: 1.2rem;
        }


        /* === Sidebar Styling === */
        .st-emotion-cache-1lcbmhc {
             background-color: var(--bg-color-sidebar);
             border-right: 1px solid var(--border-color);
             box-shadow: var(--shadow-sm);
             padding-top: 2rem; /* Space at top of sidebar */
        }
        .st-emotion-cache-16txtl3 {
            padding: 2rem 1.5rem;
        }
        .sidebar-header {
            text-align: center;
            margin-bottom: 3rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }
        .sidebar-icon {
            font-size: 3.5rem;
            display: block;
            margin-bottom: 0.8rem;
            color: var(--primary-color);
        }
        .sidebar-title {
            font-family: var(--font-family-heading);
            color: var(--primary-color);
            font-weight: 700;
            font-size: 1.8rem;
            margin-bottom: 0.4rem;
        }
        .sidebar-subtitle {
            font-family: var(--font-family-body);
            font-size: 0.95rem;
            color: var(--text-color-light);
        }

        /* Sidebar Radio Buttons Styling (Enhanced) */
         div[data-testid="stRadio"] > label {
            font-family: var(--font-family-heading);
             font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 1.2rem;
             display: block;
            color: var(--text-color);
            text-transform: uppercase;
            letter-spacing: 0.03em;
         }
        div[data-testid="stRadio"] > div {
            flex-direction: column;
            gap: 0.8rem;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] {
            background-color: var(--bg-color-main) !important;
            border-radius: var(--border-radius); /* Consistent radius */
            padding: 1rem 1.5rem;
            border: 1px solid var(--border-color);
            transition: all 0.2s ease-in-out;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 1rem; /* Space between radio circle and text */
            cursor: pointer;
            box-shadow: var(--shadow-sm);
        }
         div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
            border-color: var(--primary-color);
            background-color: rgba(0, 123, 255, 0.08) !important; /* Lighter selected background */
            box-shadow: var(--shadow-md);
            color: var(--primary-color);
         }
         div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
            border-color: var(--primary-color);
            background-color: rgba(0, 123, 255, 0.05) !important;
            box-shadow: var(--shadow-sm);
         }
         /* Hide native radio circle */
         div[data-testid="stRadio"] label[data-baseweb="radio"] span:first-child {
            display: none;
         }
         /* Custom radio circle */
         div[data-testid="stRadio"] label[data-baseweb="radio"] span:last-child::before {
            content: '';
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid var(--text-color-light);
            border-radius: 50%;
            margin-right: 10px;
            vertical-align: middle;
            transition: all 0.2s ease-in-out;
         }
         div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) span:last-child::before {
            background-color: var(--primary-color);
            border-color: var(--primary-color);
            box-shadow: 0 0 0 4px rgba(0, 123, 255, 0.2); /* Halo effect */
         }


        /* === Dividers === */
        hr.styled-divider {
            border: none;
            height: 2px; /* Thicker divider */
            background-image: linear-gradient(to right, transparent, rgba(0, 123, 255, 0.2), transparent);
            margin: 4rem 0; /* More margin */
        }

        /* === Tips Box Styling === */
        .tips-box {
            padding: 0 0.8rem;
        }
        .tips-box ul { margin: 0; padding: 0; list-style: none; }
        .tips-box li {
            margin-bottom: 0.9rem;
            position: relative;
            padding-left: 2rem;
            font-size: 0.95rem;
            color: var(--text-color-light);
            line-height: 1.6;
        }
        .tips-box li::before {
            content: '💡';
            position: absolute;
            left: 0;
            top: 0px; /* Adjust vertical alignment */
            font-size: 1.1rem;
            color: var(--accent-color);
        }
        div[data-testid="stExpander"] summary {
            font-family: var(--font-family-heading);
            font-size: 1.15rem; /* Larger */
            font-weight: 600;
            color: var(--primary-color); /* Highlight with primary color */
            padding: 0.8rem 1rem;
            border-radius: var(--border-radius);
            background-color: rgba(0, 123, 255, 0.05); /* Light background for expander summary */
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
            transition: background-color 0.2s ease;
        }
        div[data-testid="stExpander"] summary:hover {
            background-color: rgba(0, 123, 255, 0.1);
        }
        div[data-testid="stExpander"] details {
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            background-color: #ffffff;
            padding: 1.5rem; /* More padding inside */
            box-shadow: var(--shadow-md);
            margin-top: 1rem; /* Space between summary and content */
        }

        /* === Footer Styling === */
        .footer {
            text-align: center;
            color: #999999;
            padding: 4rem 0 2rem 0;
            font-size: 0.8rem;
            border-top: 1px solid var(--border-color);
            margin-top: 6rem;
            background-color: var(--bg-color-card);
        }

        /* === Mode Header Styling === */
        .mode-header {
            font-family: var(--font-family-heading);
            font-size: 3rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 3rem;
            text-align: center;
            position: relative;
            padding-bottom: 1rem;
        }
        .mode-header::after {
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


        /* === Info Boxes === */
        div[data-testid="stAlert"] {
            border-radius: var(--border-radius) !important;
            border: none !important;
            border-left: 6px solid !important; /* Thicker border */
            box-shadow: var(--shadow-md) !important;
            padding: 1.5rem 2rem !important; /* More padding */
            margin-bottom: 2rem;
            font-family: var(--font-family-body);
        }
        div[data-testid="stAlert"] strong { font-weight: 700; color: var(--text-color); }
        div[data-testid="stAlert"][kind="info"] { border-left-color: var(--primary-color) !important; background-color: #e6f2ff !important; color: #0056b3 !important;}
        div[data-testid="stAlert"][kind="success"] { border-left-color: var(--secondary-color) !important; background-color: #e6ffe6 !important; color: #1e7e34 !important;}
        div[data-testid="stAlert"][kind="warning"] { border-left-color: var(--accent-color) !important; background-color: #fffde6 !important; color: #cc9900 !important;}
        div[data-testid="stAlert"][kind="error"] { border-left-color: #dc3545 !important; background-color: #ffe6e6 !important; color: #8e1e2f !important;}


        /* === Content Boxes === */
        .content-box {
            background: var(--bg-color-card);
            padding: 3rem; /* Generous padding */
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-md);
            height: 100%;
            margin-bottom: 2rem;
        }
        .content-box h3 {
            font-family: var(--font-family-heading);
            margin-top: 0;
            margin-bottom: 1.8rem;
            color: var(--primary-color);
            font-weight: 700;
            font-size: 1.6rem;
        }
        .content-box h4 {
            font-family: var(--font-family-heading);
            margin-top: 2rem;
            margin-bottom: 1rem;
            color: var(--text-color);
            font-weight: 600;
            font-size: 1.2rem;
        }
        .content-box ol, .content-box ul { padding-left: 2rem; margin-bottom: 1.5rem;}
        .content-box li {
            margin-bottom: 1rem;
            line-height: 1.8;
            color: var(--text-color-light);
            font-family: var(--font-family-body);
        }
        .content-box li strong {
            color: var(--text-color);
            font-weight: 700;
        }
        .cta-box-content {
            display: flex; flex-direction: column; justify-content: center;
            align-items: center; text-align: center; height: 100%;
        }
        .cta-box-content p {
            font-family: var(--font-family-body);
            font-size: 1.25rem;
            color: var(--text-color-light);
            margin-bottom: 2.5rem;
            max-width: 400px;
            line-height: 1.7;
        }

        /* Utility classes */
        .text-center { text-align: center; }
        .spacer-sm { margin-bottom: 1.5rem; }
        .spacer-md { margin-bottom: 2.5rem; }
        .spacer-lg { margin-bottom: 4rem; }
        .spacer-xl { margin-bottom: 5rem; }

        </style>
    """, unsafe_allow_html=True)

    # --- Sidebar (Consistent Navigation) ---
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-header">
                <span class="sidebar-icon">🏋️</span>
                <div class="sidebar-title">FormFit AI</div>
                <div class="sidebar-subtitle">Elite Performance Training</div>
            </div>
        """, unsafe_allow_html=True)

        mode_options = ['Real-Time Analysis', 'AI Fitness Chatbot']
        current_selection_index = 0
        if st.session_state.current_view == 'chatbot':
            current_selection_index = 1
        elif st.session_state.current_view == 'landing':
             pass # Landing page doesn't strictly align with one radio option initially

        selected_mode = st.radio(
            'NAVIGATION', # Uppercase for professional look
            mode_options,
            key='sidebar_mode_select',
            index=current_selection_index,
        )

        if selected_mode == 'Real-Time Analysis' and st.session_state.current_view != 'realtime':
            set_view('realtime')
            st.rerun()
        elif selected_mode == 'AI Fitness Chatbot' and st.session_state.current_view != 'chatbot':
            set_view('chatbot')
            st.rerun()

        st.markdown('<hr class="styled-divider" style="margin: 2rem 0;">', unsafe_allow_html=True)

        tips_expanded = st.session_state.current_view == 'realtime'
        with st.expander("💡 **QUICK SETUP TIPS**", expanded=tips_expanded): # Uppercase
             st.markdown("""
                <div class="tips-box">
                    <ul>
                        <li><strong>Lighting:</strong> Ensure bright, even lighting. Avoid harsh shadows or backlighting.</li>
                        <li><strong>Visibility:</strong> Your full body should be clearly visible in the camera frame throughout the exercise.</li>
                        <li><strong>Clothing:</strong> Wear form-fitting clothing that contrasts with your background.</li>
                        <li><strong>Space:</strong> Make sure you have ample clear space around you for safe movement.</li>
                        <li><strong>Distance:</strong> Stand approximately 6-8 feet (2-2.5 meters) away from the camera for optimal tracking.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
        if st.button("↩️ BACK TO INTRODUCTION", use_container_width=True): # Uppercase
            set_view('landing')
            st.rerun()

    # --- Main Content Area --- Based on View State ---

    # == LANDING VIEW ==
    if st.session_state.current_view == 'landing':
        st.markdown('<div class="landing-container">', unsafe_allow_html=True)

        # Hero Section
        st.markdown("""
            <div class="landing-hero">
                <h1 class="main-header">Elevate Your Workout with AI</h1>
                <p class="subtitle">Perfect your form, track your progress, and gain personalized insights with our cutting-edge AI-powered fitness companion.</p>
            </div>
            """, unsafe_allow_html=True)

        # Mode Selection Section
        st.markdown('<div class="section-header">Discover Your Path to Fitness Excellence</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("""
                <div class="mode-card">
                    <div class="mode-icon">💪</div>
                    <div class="mode-title">Real-Time Form Analysis</div>
                    <div class="mode-description">
                        Receive immediate, precise feedback on your exercise technique. Our AI monitors your movements live to ensure safety and maximize effectiveness.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Start Live Analysis", key="land_to_realtime", use_container_width=True):
                set_view('realtime')
                st.rerun()

        with col2:
            st.markdown("""
                <div class="mode-card">
                    <div class="mode-icon">🗣️</div>
                    <div class="mode-title">Intelligent Fitness Chatbot</div>
                    <div class="mode-description">
                        Your personal AI coach is ready to answer questions, provide workout strategies, explain concepts, and offer tailored fitness advice 24/7.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Engage AI Coach", key="land_to_chatbot", use_container_width=True):
                set_view('chatbot')
                st.rerun()

        st.markdown('<hr class="styled-divider spacer-xl">', unsafe_allow_html=True)

        # Core Features Highlight
        st.markdown('<div class="section-header">Unlock Your Potential with Our Core Features</div>', unsafe_allow_html=True)
        feature_cols = st.columns(4, gap="large") # Increased gap
        landing_features = [
            {"icon": "🤖", "title": "Precision AI Tracking", "desc": "Utilizes advanced computer vision for highly accurate and reliable movement analysis."},
            {"icon": "📈", "title": "Comprehensive Performance Metrics", "desc": "Track reps, sets, tempo, and estimated fatigue to fine-tune your training regimens."},
            {"icon": "✅", "title": "Actionable Form Correction", "desc": "Get clear, instant guidance to correct your form, preventing injuries and boosting results."},
            {"icon": "🧠", "title": "Personalized Fitness Intelligence", "desc": "Access a wealth of knowledge and bespoke advice from your AI fitness companion."},
        ]
        for i, feature in enumerate(landing_features):
            with feature_cols[i]:
                 st.markdown(f"""
                    <div class="feature-card">
                        <div class="feature-icon">{feature['icon']}</div>
                        <div class="feature-title">{feature['title']}</div>
                        <div class="feature-description">{feature['desc']}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


    # == CHATBOT VIEW ==
    elif st.session_state.current_view == 'chatbot':
        st.markdown('<div class="mode-header">Your AI Fitness Coach</div>', unsafe_allow_html=True)
        st.info("""
            **Empower your fitness journey with personalized advice from our AI coach.**
            Ask about:
            * **Exercise Techniques:** Master proper form for any movement.
            * **Workout Strategies:** Design effective routines and optimize your training.
            * **Injury Prevention:** Learn how to stay safe and recover smarter.
            * **Nutrition & Wellness:** Get insights into healthy eating and holistic fitness.
        """, icon="💬")
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
        chat_ui()


    # == REAL-TIME ANALYSIS VIEW ==
    elif st.session_state.current_view == 'realtime':
        st.markdown('<div class="mode-header">Real-Time Performance Analysis</div>', unsafe_allow_html=True)

        st.markdown("<h4 class='text-center' style='font-family: var(--font-family-heading); font-weight: 700; font-size: 1.8rem; margin-bottom: 2.5rem; color: var(--primary-color);'>How Live Analysis Transforms Your Training:</h4>", unsafe_allow_html=True)
        cols_features = st.columns(4, gap="large")
        feature_data_realtime = [
            {"icon": "👁️", "title": "Automatic Exercise Recognition", "desc": "Intelligently identifies Push-ups, Squats, Bicep Curls, and Shoulder Presses as you begin."},
            {"icon": "⚙️", "title": "Dynamic Joint Angle Tracking", "desc": "Analyzes precise joint angles in real-time, guiding you through optimal range of motion."},
            {"icon": "🔢", "title": "Accurate Repetition Counting", "desc": "Utilizes advanced algorithms to count your reps flawlessly, ensuring precise workout logs."},
            {"icon": "⚡", "title": "Real-Time Fatigue Assessment", "desc": "Monitors changes in your movement patterns to provide insights into your performance and fatigue levels."},
        ]
        num_features = len(feature_data_realtime)
        cols_needed = st.columns(num_features, gap="large")

        for i, feature in enumerate(feature_data_realtime):
            with cols_needed[i]:
                st.markdown(f"""
                    <div class="feature-card">
                        <div class="feature-icon">{feature['icon']}</div>
                        <div class="feature-title">{feature['title']}</div>
                        <div class="feature-description">{feature['desc']}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown('<hr class="styled-divider spacer-xl">', unsafe_allow_html=True)

        col_instructions, col_cta = st.columns([6, 4], gap="large")

        with col_instructions:
            st.markdown("""
                <div class="content-box">
                    <h3>Initiate Your Live Training Session:</h3>
                    <ol>
                        <li><strong>Optimal Environment Setup:</strong> Refer to the 'Quick Setup Tips' in the sidebar to ensure ideal camera positioning and lighting conditions.</li>
                        <li><strong>Position Yourself:</strong> Stand or place yourself within the camera frame, ensuring your entire body is visible and unobstructed.</li>
                        <li><strong>Begin Exercise:</strong> Start performing one of the automatically recognized exercises (Push-up, Squat, Bicep Curl, or Shoulder Press).</li>
                        <li><strong>Engage with Real-Time Feedback:</strong> The system will provide:
                            <ul>
                                <li><strong>Visual Cues:</strong> On-screen guidance for form correction and positive reinforcement.</li>
                                <li><strong>Rep/Set Tracking:</strong> Automated counting of valid repetitions and completed sets.</li>
                                <li><strong>Fatigue Monitoring:</strong> Live estimation of your current performance and fatigue status.</li>
                            </ul>
                        </li>
                        <li><strong>Conclude Session:</strong> Click the 'Stop Session' button, which will appear below the camera feed, when you wish to end the analysis.</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)

        with col_cta:
            st.markdown("""
                <div class="content-box cta-box-content">
                    <h3>Ready for Smarter Workouts?</h3>
                    <p>Activate your camera and experience the future of personalized fitness coaching, directly from your home.</p>
                </div>
            """, unsafe_allow_html=True)
            start_button = st.button('🚀 LAUNCH AI ANALYSIS SESSION', use_container_width=True, key="start_analysis", type="primary")

        if start_button:
            col_instructions.empty()
            col_cta.empty()
            st.success("Initializing Analysis... Please ensure your camera is active and you are in frame. Get Ready!", icon="⏳")
            time.sleep(2) # Increased pause for better user experience

            st.markdown('<hr class="styled-divider" style="margin-top: 0rem;">', unsafe_allow_html=True)

            try:
                analysis_placeholder = st.empty()
                with analysis_placeholder.container():
                    st.info("Live analysis in progress. Perform your exercises clearly in view.", icon="📹")
                    exer = exercise.Exercise()
                    exer.auto_classify_exercise()

                st.success("Analysis Session Concluded. Great Workout!", icon="✅")

            except AttributeError as ae:
                 st.error(f"Initialization Error: A required component might be missing. Please check the setup. ({ae})")
            except ImportError as ie:
                 st.error(f"Import Error: A required library is missing. Please ensure all dependencies (like OpenCV, Mediapipe) are installed. ({ie})")
            except Exception as e:
                 st.error(f"An unexpected error occurred during analysis: {e}")
                 st.exception(e)
                 st.warning("Troubleshooting: Ensure camera access is allowed, required libraries are installed, and try restarting the application.")


    # --- Footer (Always Visible) ---
    st.markdown("""
        <div class="footer">
             FormFit AI © 2025 | Developed by Shubh Jain | Empowering Your Fitness Journey
        </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()

#--- END OF FILE main.py ---