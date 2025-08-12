import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt
from io import BytesIO
import random
import colorsys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import plotly.figure_factory as ff
import io

def format_number(n, decimals=0, is_currency=False):
    """
    Format a number with commas and optional decimals.
    
    Args:
        n: The number to format
        decimals: Number of decimal places to show (default: 0)
        is_currency: If True, adds 'Kshs ' prefix (default: False)
        
    Returns:
        str: Formatted number as string
        
    Examples:
        format_number(12345) -> '12,345'
        format_number(12345.67, 2) -> '12,345.67'
        format_number(12345.67, 2, True) -> 'Kshs 12,345.67'
    """
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "-"
    try:
        formatted = f"{n:,.{decimals}f}" if decimals > 0 else f"{int(n):,}"
        return f"Kshs {formatted}" if is_currency else formatted
    except (ValueError, TypeError):
        return str(n)

def format_hog_id(hog_id):
    """
    Format a Hog ID as a zero-padded three-digit string. Returns '-' if not a valid integer.
    """
    try:
        return f"{int(hog_id):03d}"
    except (ValueError, TypeError):
        return "-"

DATA_FILE = 'hog_data.csv'
FINANCIAL_DATA_FILE = 'financial_data.csv'

st.set_page_config(layout="wide", page_title="Hog Weight Tracking App", page_icon="🐖")

# --- Enhanced Mobile-Friendly UI ---
st.markdown("""
<style>
/* Responsive font and padding for mobile */
html, body, .stApp {
    font-size: 1.08em;
    padding: 0;
    margin: 0;
}
@media (max-width: 600px) {
    .stApp {
        padding: 0.5em 0.2em 0.5em 0.2em;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05em;
        padding: 8px 8px;
        min-width: 90px;
    }
    .stButton>button, .stDownloadButton>button {
        font-size: 1.1em;
        padding: 10px 18px;
    }
    .stTextInput>div>div>input {
        font-size: 1.1em;
        padding: 8px 10px;
    }
    .stSelectbox>div>div>div {
        font-size: 1.1em;
    }
    .stDataFrameContainer {
        font-size: 1.05em;
    }
}
/* Floating Action Button (FAB) for quick add */
.fab {
    position: fixed;
    bottom: 32px;
    right: 32px;
    z-index: 1000;
}
.fab-btn {
    background: #FF4B4B;
    color: #fff;
    border: none;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    font-size: 2.2em;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    cursor: pointer;
    transition: background 0.2s;
}
.fab-btn:hover {
    background: #d13a3a;
}
</style>
""", unsafe_allow_html=True)

# Interactive sidebar toggle for mobile
if 'sidebar_open' not in st.session_state:
    st.session_state['sidebar_open'] = True

def toggle_sidebar():
    st.session_state['sidebar_open'] = not st.session_state['sidebar_open']







# Custom CSS for tabs and overall styling
st.markdown("""
<style>
/* General Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px; /* Adjust space between tabs */
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #f0f2f6; /* Light gray for inactive tabs */
    border-radius: 4px 4px 0 0;
    gap: 10px;
    padding-top: 10px;
    padding-bottom: 10px;
    padding-left: 20px;
    padding-right: 20px;
}

.stTabs [aria-selected="true"] {
    background-color: #0E1117; /* Dark background for active tab */
    color: white; /* White text for active tab */
    font-weight: bold;
    border-bottom: 3px solid #FF4B4B; /* Highlight for active tab */
}

/* Hover effect for tabs */
.stTabs [data-baseweb="tab"]:hover {
    background-color: #e0e2e6; /* Slightly darker gray on hover */
    color: #333333; /* Darker text on hover */
}

/* Overall app background for a cohesive look */
.stApp {
    background-color: #FFFFFF; /* White background for the main app */
}

/* Adjust header spacing */
h1, h2, h3, h4, h5, h6 {
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

</style>
""", unsafe_allow_html=True)

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df['Date'] = pd.to_datetime(df['Date']).dt.date # Ensure date format consistency
        return df
    return pd.DataFrame(columns=['Hog ID', 'Date', 'Weight (kg)'])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def load_financial_data():
    if os.path.exists(FINANCIAL_DATA_FILE):
        df = pd.read_csv(FINANCIAL_DATA_FILE)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        # Ensure all expected columns exist, fill missing with NaN
        for col in ['Type', 'Category', 'Description', 'Hog ID', 'Weight (kg)', 'Price/kg', 'Amount']:
            if col not in df.columns:
                df[col] = pd.NA
        return df
    return pd.DataFrame(columns=['Date', 'Type', 'Category', 'Description', 'Hog ID', 'Weight (kg)', 'Price/kg', 'Amount'])

def save_financial_data(df):
    df.to_csv(FINANCIAL_DATA_FILE, index=False)

def no_growth_summary(no_gain_data):
    """
    Generate a summary of hogs with no weight gain.
    
    Args:
        no_gain_data: DataFrame containing hogs with no weight gain
        
    Returns:
        DataFrame: Summary of hogs with no weight gain
    """
    if no_gain_data.empty:
        return pd.DataFrame()
        
    # Group by Hog ID and get the latest record for each hog
    summary = no_gain_data.sort_values('Date').groupby('Hog ID').last().reset_index()
    
    # Calculate weeks without gain for each hog
    weeks_no_gain = no_gain_data.groupby('Hog ID').size().reset_index(name='Weeks Without Gain')
    
    # Merge the data
    summary = summary.merge(weeks_no_gain, on='Hog ID', how='left')
    
    # Select and rename columns for display
    summary = summary[['Hog ID', 'Weight (kg)', 'Growth (%)', 'Weeks Without Gain']]
    
    return summary


def generate_summary_report_text(summary_report):
    """Generate a text version of the summary report."""
    report = []
    report.append("=== Hog Farm Performance Report ===")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    # Overall Summary
    report.append("OVERALL FARM SUMMARY")
    report.append("-" * 30)
    report.append(f"Total Hogs Tracked: {summary_report['overall']['total_hogs']}")
    
    if 'average_daily_gain' in summary_report['overall']:
        report.append(f"Average Daily Gain: {summary_report['overall']['average_daily_gain']:.2f} kg/day")
    
    # Performance Leaders
    report.append("\nTOP PERFORMERS")
    report.append("-" * 30)
    if 'best_overall_performer' in summary_report['overall']:
        best = summary_report['overall']['best_overall_performer']
        report.append(f"Best Performer: Hog #{best['id']} ({best['adg']:.2f} kg/day)")
    
    if 'least_overall_performer' in summary_report['overall']:
        least = summary_report['overall']['least_overall_performer']
        report.append(f"Needs Attention: Hog #{least['id']} ({least['adg']:.2f} kg/day)")
    
    # Latest Week Summary
    report.append("\nLATEST WEEK SUMMARY")
    report.append("-" * 30)
    report.append(f"Week Ending: {summary_report['latest_week_date']}")
    
    if 'average_herd_gain' in summary_report['latest_week'] and summary_report['latest_week']['average_herd_gain'] is not None:
        report.append(f"Average Herd Gain: {summary_report['latest_week']['average_herd_gain']:.1f} kg")
    
    if 'highest_performer' in summary_report['latest_week'] and summary_report['latest_week']['highest_performer']:
        hp = summary_report['latest_week']['highest_performer']
        report.append(f"Top Gainer: Hog #{hp['id']} (+{hp['weight_change']:.1f} kg, {hp['growth_percent']:.1f}%)")
    
    if 'least_performer' in summary_report['latest_week'] and summary_report['latest_week']['least_performer']:
        lp = summary_report['latest_week']['least_performer']
        report.append(f"Lowest Gainer: Hog #{lp['id']} ({lp['weight_change']:.1f} kg, {lp['growth_percent']:.1f}%)")
    
    # Health Alerts
    if summary_report['potential_health_concern_hogs']:
        report.append("\nHEALTH ALERTS")
        report.append("-" * 30)
        for concern in summary_report['potential_health_concern_hogs']:
            report.append(f"Hog #{concern['id']}: {concern['consecutive_measurements']} consecutive non-positive weight changes (as of {concern['latest_date']})")
    
    # Recommendations
    report.append("\nRECOMMENDATIONS")
    report.append("-" * 30)
    
    if summary_report['potential_health_concern_hogs']:
        report.append("* Immediate veterinary attention required for hogs with consecutive non-positive weight changes")
    
    if summary_report['latest_week'].get('no_growth_hogs'):
        report.append("* Investigate feeding and environment for hogs with no growth")
    
    if not any([summary_report['potential_health_concern_hogs'], 
               summary_report['latest_week'].get('no_growth_hogs')]):
        report.append("* All hogs are showing normal growth patterns. Continue current care regimen.")
    
    return "\n".join(report)


def generate_summary_report(display_data, all_growth_data):
    report_summary = {}

    if display_data.empty:
        report_summary['status'] = "No data available to generate a summary report."
        return report_summary

    # Overall Summary
    total_hogs = display_data['Hog ID'].nunique()
    report_summary['overall'] = {
        'total_hogs': total_hogs
    }
    
    # Calculate Overall Average Daily Gain (ADG) for the entire herd
    overall_adg_data = []
    for hog_id in display_data['Hog ID'].unique():
        hog_records = display_data[display_data['Hog ID'] == hog_id].sort_values(by='Date')
        if len(hog_records) > 1:
            # Calculate difference in days and weight for each pair of consecutive measurements
            time_diff = (hog_records['Date'].iloc[-1] - hog_records['Date'].iloc[0]).days
            weight_diff = hog_records['Weight (kg)'].iloc[-1] - hog_records['Weight (kg)'].iloc[0]
            
            if time_diff > 0:
                adg = weight_diff / time_diff
                overall_adg_data.append({'Hog ID': hog_id, 'ADG': adg})

    if overall_adg_data:
        overall_adg_df = pd.DataFrame(overall_adg_data)
        report_summary['overall']['average_daily_gain'] = overall_adg_df['ADG'].mean()
        
        # Overall Best and Least Performers based on ADG
        best_overall_adg = overall_adg_df.loc[overall_adg_df['ADG'].idxmax()]
        report_summary['overall']['best_overall_performer'] = {
            'id': int(best_overall_adg['Hog ID']),
            'adg': best_overall_adg['ADG']
        }

        least_overall_adg = overall_adg_df.loc[overall_adg_df['ADG'].idxmin()]
        report_summary['overall']['least_overall_performer'] = {
            'id': int(least_overall_adg['Hog ID']),
            'adg': least_overall_adg['ADG']
        }


    # Latest Week Summary
    latest_date = display_data['Date'].max()
    report_summary['latest_week_date'] = latest_date.strftime('%d/%m/%Y')
    
    latest_week_data = display_data[display_data['Date'] == latest_date]

    report_summary['latest_week'] = {}

    if all_growth_data.empty:
        report_summary['latest_week']['growth_status'] = "No growth data available for the latest week summary."
    else:
        all_growth_data_copy = all_growth_data.copy()
        all_growth_data_copy['Date'] = pd.to_datetime(all_growth_data_copy['Date'])
        
        latest_growth_date = all_growth_data_copy['Date'].max()
        latest_week_growth_data = all_growth_data_copy[all_growth_data_copy['Date'] == latest_growth_date]

        if not latest_week_growth_data.empty:
            # Average herd gain for latest week
            average_herd_gain = latest_week_growth_data['Weight Change (kg)'].mean()
            if pd.notna(average_herd_gain):
                report_summary['latest_week']['average_herd_gain'] = average_herd_gain
            else:
                report_summary['latest_week']['average_herd_gain'] = None

            # Highest performer for latest week
            if not latest_week_growth_data['Growth (%)'].empty:
                highest_performer = latest_week_growth_data.loc[latest_week_growth_data['Growth (%)'].idxmax()]
                report_summary['latest_week']['highest_performer'] = {
                    'id': int(highest_performer['Hog ID']),
                    'growth_percent': highest_performer['Growth (%)'],
                    'weight_change': highest_performer['Weight Change (kg)'] # Add weight change
                }
            else:
                report_summary['latest_week']['highest_performer'] = None

            # Least performer (most negative growth/loss) for latest week
            if not latest_week_growth_data['Weight Change (kg)'].empty:
                least_performer = latest_week_growth_data.loc[latest_week_growth_data['Weight Change (kg)'].idxmin()]
                report_summary['latest_week']['least_performer'] = {
                    'id': int(least_performer['Hog ID']),
                    'weight_change': least_performer['Weight Change (kg)'],
                    'growth_percent': least_performer['Growth (%)']
                }
            else:
                report_summary['latest_week']['least_performer'] = None

            # Hogs with no growth for latest week
            no_growth_hogs = latest_week_growth_data[latest_week_growth_data['Weight Change (kg)'] <= 0]
            if not no_growth_hogs.empty:
                report_summary['latest_week']['no_growth_hogs'] = no_growth_hogs['Hog ID'].astype(int).unique().tolist()
            else:
                report_summary['latest_week']['no_growth_hogs'] = []

        else:
            report_summary['latest_week']['growth_status'] = "No recent growth data to generate a detailed summary for the latest week."

    # Hogs with Potential Health Concerns (e.g., 3+ consecutive non-positive weight changes)
    potential_health_concern_hogs = []
    if not all_growth_data.empty:
        for hog_id in all_growth_data['Hog ID'].unique():
            hog_growth_data = all_growth_data[all_growth_data['Hog ID'] == hog_id].sort_values(by='Date')
            
            consecutive_non_positive = 0
            for i in range(len(hog_growth_data)):
                if hog_growth_data['Weight Change (kg)'].iloc[i] <= 0:
                    consecutive_non_positive += 1
                else:
                    consecutive_non_positive = 0
                
                if consecutive_non_positive >= 3:
                    latest_date_of_concern = hog_growth_data['Date'].iloc[i]
                    potential_health_concern_hogs.append({
                        'id': int(hog_id),
                        'latest_date': latest_date_of_concern.strftime('%d/%m/%Y'),
                        'consecutive_measurements': consecutive_non_positive
                    })
                    break # Report once per hog

    report_summary['potential_health_concern_hogs'] = potential_health_concern_hogs

    return report_summary

def main():
    # --- Mobile-Friendly UI: Set wide layout and responsive design ---
    st.set_page_config(layout="wide", page_title="Hog Weight Tracking App", page_icon="🐖")

    # --- User Authentication & Roles ---
    if 'user_authenticated' not in st.session_state:
        st.session_state['user_authenticated'] = False
    if 'user_role' not in st.session_state:
        st.session_state['user_role'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None

    # User database (for demo; replace with secure storage in production)
    if 'USER_DB' not in st.session_state:
        st.session_state['USER_DB'] = {
            'admin': {'password': 'admin123', 'role': 'admin'},
            'staff': {'password': 'staff123', 'role': 'staff'},
            'steve': {'password': 'steve123', 'role': 'admin'},
            'vince': {'password': 'vince123', 'role': 'admin'},
            'viewer': {'password': 'viewer123', 'role': 'viewer'}
        }
    USER_DB = st.session_state['USER_DB']

    def login_form():
        # Custom CSS for the login form
        st.markdown("""
        <style>
            .login-container {
                max-width: 450px;
                margin: 0 auto;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                background-color: white;
                border: 1px solid #e0e0e0;
            }
            .login-header {
                text-align: center;
                margin-bottom: 1.5rem;
            }
            .login-header h1 {
                color: #2c3e50;
                margin-bottom: 0.5rem;
            }
            .login-header p {
                color: #7f8c8d;
                margin-top: 0;
            }
            .stTextInput>div>div>input {
                border-radius: 8px;
                padding: 0.75rem;
                border: 1px solid #ddd;
            }
            .stTextInput>div>div>input:focus {
                border-color: #3498db;
                box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
            }
            .stButton>button {
                width: 100%;
                border-radius: 8px;
                padding: 0.75rem;
                font-weight: 600;
                background-color: #2ecc71;
                border: none;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #27ae60;
                transform: translateY(-1px);
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Login container with custom styling
        with st.container():
            st.markdown("""
            <div class='login-container'>
                <div class='login-header'>
                    <h1>🐷 Hog Weight Tracker</h1>
                    <p>Enter your credentials to access the dashboard</p>
                </div>
            """, unsafe_allow_html=True)
            
            with st.form("login_form"):
                # Username field
                username = st.text_input(
                    "👤 Username",
                    placeholder="Enter your username",
                    key="login_username"
                )
                
                # Password field
                password = st.text_input(
                    "🔒 Password",
                    type="password",
                    placeholder="Enter your password",
                    key="login_password"
                )
                
                # Login button
                submit = st.form_submit_button("Login to Dashboard")
                
                if submit:
                    if username in USER_DB and password == USER_DB[username]['password']:
                        # Set session state
                        st.session_state['user_authenticated'] = True
                        st.session_state['user_role'] = USER_DB[username]['role']
                        st.session_state['username'] = username
                        
                        # Show success toast
                        st.toast(f"👋 Welcome back, {username}!", icon='✅')
                        st.rerun()
                    else:
                        # Show error message with better styling
                        st.error("❌ Invalid username or password. Please try again.")
                        
                        # Add a subtle animation to the error message
                        st.markdown("""
                        <style>
                            @keyframes shake {
                                0%, 100% { transform: translateX(0); }
                                10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
                                20%, 40%, 60%, 80% { transform: translateX(5px); }
                            }
                            .stAlert {
                                animation: shake 0.5s ease-in-out;
                            }
                        </style>
                        """, unsafe_allow_html=True)
            
            # Footer with app version or additional info
            st.markdown("""
                <div style='text-align: center; margin-top: 2rem; color: #95a5a6; font-size: 0.9rem;'>
                    Hog Weight Tracking App v1.0.0
                </div>
            </div>
            """, unsafe_allow_html=True)
    # --- Admin User Management ---
    # (Handled only in sidebar layout below to avoid duplicate form keys)

    def logout():
        st.session_state['user_authenticated'] = False
        st.session_state['user_role'] = None
        st.session_state['username'] = None
        st.rerun()

    if not st.session_state['user_authenticated']:
        login_form()
        return

    # Robust error handling for data loading (must be before sidebar uses hog_data/financial_data)
    try:
        if 'hog_data' not in st.session_state:
            st.session_state['hog_data'] = load_data()
    except Exception as e:
        st.session_state['hog_data'] = pd.DataFrame(columns=['Hog ID', 'Date', 'Weight (kg)'])
        st.error(f"Error loading hog data: {e}")

    try:
        if 'financial_data' not in st.session_state:
            st.session_state['financial_data'] = load_financial_data()
    except Exception as e:
        st.session_state['financial_data'] = pd.DataFrame(columns=['Date', 'Type', 'Category', 'Description', 'Hog ID', 'Weight (kg)', 'Price/kg', 'Amount'])
        st.error(f"Error loading financial data: {e}")

    # --- Sidebar Layout ---
    with st.sidebar:
        # App Header
        st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='font-size: 24px; margin-bottom: 5px;'>🐷 Hog Weight Tracker</h1>
            <div style='height: 2px; background: linear-gradient(90deg, transparent, #ddd, transparent); margin: 10px 0;'></div>
        </div>
        """, unsafe_allow_html=True)
        
        # User Info & Logout (always at top)
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 10px 15px; border-radius: 8px; margin-bottom: 8px;'>
            <div style='font-weight: 600; color: #1f1f1f;'>{st.session_state['username']}</div>
            <div style='font-size: 0.8em; color: #666; margin-bottom: 8px;'>{st.session_state['user_role'].capitalize()} User</div>
            <div style='display: flex; justify-content: center;'>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", key="sidebar_logout_btn", type='primary', use_container_width=True):
            logout()
            
        st.markdown("</div></div>", unsafe_allow_html=True)

        # Admin User Management (admin only)
        if st.session_state.get('user_role') == 'admin':
            with st.expander("👑 Admin Panel", expanded=False):
                st.caption("Add New User")
                with st.form("add_user_form_sidebar"):
                    new_username = st.text_input("New Username", key="add_user_username_sidebar")
                    new_password = st.text_input("New Password", type="password", key="add_user_password_sidebar")
                    new_role = st.selectbox("Role", ["admin", "staff", "viewer"], key="add_user_role_sidebar")
                    submit_new_user = st.form_submit_button("Add User")
                    if submit_new_user:
                        if not new_username or not new_password:
                            st.error("Username and password are required.")
                        elif new_username in USER_DB:
                            st.error("Username already exists.")
                        else:
                            st.session_state['USER_DB'][new_username] = {"password": new_password, "role": new_role}
                            st.success(f"User '{new_username}' added as {new_role}.")
                st.caption("Current Users")
                user_table = [
                    {"Username": u, "Role": v["role"]} for u, v in USER_DB.items()
                ]
                st.dataframe(user_table, use_container_width=True, hide_index=True)

        # Hog Management Section
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        with st.expander("🐷 Hog Management", expanded=False):
            hog_data_display = st.session_state['hog_data'].copy()
            if 'Hog ID' in hog_data_display.columns:
                hog_data_display['Hog ID'] = hog_data_display['Hog ID'].apply(format_hog_id)
            formatted_hog_ids = [format_hog_id(hid) for hid in st.session_state['hog_data']['Hog ID'].unique() if pd.notna(hid)]
            st.caption(f"All Hogs: {', '.join(formatted_hog_ids) if formatted_hog_ids else 'None'}")
            if st.session_state['user_role'] in ['admin', 'staff']:
                hog_id = st.text_input("Add Hog ID:", "", key='add_hog_id_input_sidebar')
                add_hog_placeholder = st.empty()
                if add_hog_placeholder.button("Add Hog", key='add_hog_button_sidebar'):
                    if not hog_id:
                        add_hog_placeholder.error("❌ Please enter a Hog ID.")
                    else:
                        try:
                            hog_id_int = int(hog_id)
                            if hog_id_int < 0:
                                st.error("❌ Hog ID cannot be negative.")
                            elif hog_id_int in st.session_state['hog_data']['Hog ID'].dropna().astype(int).values:
                                st.error(f"❌ Hog {hog_id_int:03d} already exists.")
                            else:
                                st.session_state['confirm_add_hog_sidebar'] = True
                                st.session_state['hog_id_to_add_sidebar'] = hog_id_int
                                st.rerun()
                        except ValueError:
                            st.error("❌ Hog ID must be a valid integer.")

                if st.session_state.get('confirm_add_hog_sidebar', False):
                    with add_hog_placeholder.container():
                        st.warning(f"Add Hog: {int(st.session_state['hog_id_to_add_sidebar']):03d}?")
                        col_confirm_add_hog_yes, col_confirm_add_hog_no = st.columns(2)
                        with col_confirm_add_hog_yes:
                            if st.button("Yes", key='confirm_add_hog_yes_sidebar'):
                                try:
                                    new_hog_df = pd.DataFrame([{'Hog ID': int(st.session_state['hog_id_to_add_sidebar']), 'Date': None, 'Weight (kg)': None}])
                                    st.session_state['hog_data'] = pd.concat([st.session_state['hog_data'], new_hog_df], ignore_index=True)
                                    save_data(st.session_state['hog_data'])
                                    st.success(f"Hog {int(st.session_state['hog_id_to_add_sidebar']):03d} added!")
                                except Exception as e:
                                    st.error(f"Error adding hog: {e}")
                                finally:
                                    del st.session_state['confirm_add_hog_sidebar']
                                    del st.session_state['hog_id_to_add_sidebar']
                                    st.rerun()
                        with col_confirm_add_hog_no:
                            if st.button("Cancel", key='confirm_add_hog_no_sidebar'):
                                del st.session_state['confirm_add_hog_sidebar']
                                del st.session_state['hog_id_to_add_sidebar']
                                st.info("Hog addition cancelled.")
                                st.rerun()

                formatted_hog_ids = [format_hog_id(hid) for hid in st.session_state['hog_data']['Hog ID'].unique() if pd.notna(hid)]
                hog_to_remove_display = st.selectbox("Remove Hog:", formatted_hog_ids, key='remove_hog_selectbox_sidebar')
                remove_hog_placeholder = st.empty()
                if hog_to_remove_display:
                    if remove_hog_placeholder.button("Remove Selected", key='remove_hog_button_sidebar'):
                        st.session_state['confirm_remove_hog_sidebar'] = True
                        st.session_state['hogs_to_remove_display_sidebar'] = [hog_to_remove_display]
                        st.rerun()
                if st.session_state.get('confirm_remove_hog_sidebar', False):
                    with remove_hog_placeholder.container():
                        st.warning(f"Remove Hog(s): {', '.join(st.session_state['hogs_to_remove_display_sidebar'])}? This cannot be undone.")
                        col_confirm_remove_yes, col_confirm_remove_no = st.columns(2)
                        with col_confirm_remove_yes:
                            if st.button("Yes", key='confirm_remove_hog_yes_sidebar'):
                                hogs_to_remove_int = [int(hid) for hid in st.session_state['hogs_to_remove_display_sidebar']]
                                st.session_state['hog_data'] = st.session_state['hog_data'][~st.session_state['hog_data']['Hog ID'].isin(hogs_to_remove_int)]
                                st.session_state['hog_data'].dropna(subset=['Date', 'Weight (kg)'], inplace=True)
                                save_data(st.session_state['hog_data'])
                                st.success(f"Hog(s) {', '.join(st.session_state['hogs_to_remove_display_sidebar'])} removed.")
                                del st.session_state['confirm_remove_hog_sidebar']
                                del st.session_state['hogs_to_remove_display_sidebar']
                                st.rerun()
                        with col_confirm_remove_no:
                            if st.button("Cancel", key='confirm_remove_hog_no_sidebar'):
                                del st.session_state['confirm_remove_hog_sidebar']
                                del st.session_state['hogs_to_remove_display_sidebar']
                                st.info("Hog removal cancelled.")
                                st.rerun()
            else:
                st.info("View only. You can see all hogs above.")

        # Delete Weight Records Section
        with st.expander("Delete Weight Records", expanded=False):
            if st.session_state['user_role'] in ['admin', 'staff']:
                delete_date = st.date_input("Delete Records From Date:", value=None, key='delete_date_input_sidebar')
                delete_hog_ids = st.multiselect("Delete Records For Hog(s) (Optional):",
                                                st.session_state['hog_data']['Hog ID'].unique(),
                                                key='delete_hog_ids_multiselect_sidebar')
                delete_records_placeholder = st.empty()
                if delete_date or delete_hog_ids:
                    if delete_records_placeholder.button("Delete Records", key='delete_records_button_sidebar'):
                        st.session_state['confirm_delete_records_sidebar'] = True
                        st.session_state['delete_date_sidebar'] = delete_date
                        st.session_state['delete_hog_ids_sidebar'] = delete_hog_ids
                        st.rerun()
                if st.session_state.get('confirm_delete_records_sidebar', False):
                    with delete_records_placeholder.container():
                        display_delete_date = st.session_state['delete_date_sidebar'] if st.session_state['delete_date_sidebar'] else 'All Dates'
                        display_delete_hog_ids = ', '.join([str(int(hid)) for hid in st.session_state['delete_hog_ids_sidebar']]) if st.session_state['delete_hog_ids_sidebar'] else 'All Hogs'
                        st.warning(f"Delete records for Date: {display_delete_date} and Hog(s): {display_delete_hog_ids}? This cannot be undone.")
                        col_confirm_delete_yes, col_confirm_delete_no = st.columns(2)
                        with col_confirm_delete_yes:
                            if st.button("Yes", key='confirm_delete_records_yes_sidebar'):
                                initial_row_count = len(st.session_state['hog_data'])
                                df = st.session_state['hog_data'].copy()
                                if st.session_state['delete_date_sidebar']:
                                    df = df[df['Date'] != st.session_state['delete_date_sidebar']]
                                if st.session_state['delete_hog_ids_sidebar']:
                                    df = df[~df['Hog ID'].isin(st.session_state['delete_hog_ids_sidebar'])]
                                deleted_row_count = initial_row_count - len(df)
                                st.session_state['hog_data'] = df
                                save_data(st.session_state['hog_data'])
                                if deleted_row_count > 0:
                                    st.success(f"{deleted_row_count} record(s) deleted.")
                                else:
                                    st.info("No records found matching the criteria to delete.")
                                del st.session_state['confirm_delete_records_sidebar']
                                del st.session_state['delete_date_sidebar']
                                del st.session_state['delete_hog_ids_sidebar']
                                st.rerun()
                        with col_confirm_delete_no:
                            if st.button("Cancel", key='confirm_delete_records_no_sidebar'):
                                del st.session_state['confirm_delete_records_sidebar']
                                del st.session_state['delete_date_sidebar']
                                del st.session_state['delete_hog_ids_sidebar']
                                st.info("Deletion cancelled.")
                                st.rerun()
            else:
                st.warning("🔒 No permission to delete records.")

        # Filter/Search Section
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        with st.expander("🔍 Filter & Search", expanded=False):
            search_hog_id = st.text_input("Search by Hog ID:", "", key='sidebar_search_hog_id')
            search_start_date = st.date_input("Search from Date:", value=None, key='sidebar_start_date')
            search_end_date = st.date_input("Search to Date:", value=None, key='sidebar_end_date')

        # Import Data Section
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        with st.expander("📤 Import Data", expanded=False):
            if st.session_state['user_role'] not in ['admin', 'staff']:
                st.warning("🔒 No permission to import data.")
            else:
                uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key='file_uploader_sidebar')
                if uploaded_file is not None:
                    try:
                        imported_df = pd.read_excel(uploaded_file, header=0)
                        imported_df.columns = [str(col).strip() for col in imported_df.columns]
                        hog_id_col_name = next((col for col in imported_df.columns if col.strip().lower() == 'hog id'), None)
                        if hog_id_col_name and hog_id_col_name != 'Hog ID':
                            imported_df.rename(columns={hog_id_col_name: 'Hog ID'}, inplace=True)
                        is_pivoted_format = False
                        processed_columns_for_melting = []
                        if 'Hog ID' in imported_df.columns:
                            date_candidate_cols = []
                            for col in imported_df.columns:
                                if col != 'Hog ID':
                                    parsed_date = pd.to_datetime(str(col), errors='coerce', dayfirst=True)
                                    if pd.notna(parsed_date):
                                        date_candidate_cols.append(col)
                            if len(date_candidate_cols) >= 1:
                                is_pivoted_format = True
                                processed_columns_for_melting = date_candidate_cols
                        if is_pivoted_format:
                            id_vars = ['Hog ID']
                            value_vars = processed_columns_for_melting
                            imported_df_melted = imported_df.melt(id_vars=id_vars, var_name='Date', value_name='Weight (kg)', value_vars=value_vars)
                            def parse_date_robust(date_val):
                                if isinstance(date_val, pd.Timestamp):
                                    return date_val.date()
                                if isinstance(date_val, dt.date):
                                    return date_val
                                date_str = str(date_val).strip()
                                try:
                                    parsed = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
                                    if pd.notna(parsed):
                                        return parsed.date()
                                except Exception:
                                    pass
                                try:
                                    parsed = pd.to_datetime(date_str, errors='coerce')
                                    if pd.notna(parsed):
                                        return parsed.date()
                                except Exception:
                                    pass
                                return pd.NaT
                            imported_df_melted['Date'] = imported_df_melted['Date'].apply(parse_date_robust)
                            imported_df = imported_df_melted.dropna(subset=['Date', 'Weight (kg)'])
                            imported_df = imported_df[imported_df['Date'].notna()]
                        if all(col in imported_df.columns for col in ['Hog ID', 'Date', 'Weight (kg)']):
                            def ensure_date_only(val):
                                if isinstance(val, pd.Timestamp):
                                    return val.date()
                                if isinstance(val, dt.date):
                                    return val
                                try:
                                    return pd.to_datetime(val, dayfirst=True).date()
                                except:
                                    return pd.NaT
                            imported_df['Date'] = imported_df['Date'].apply(ensure_date_only)
                            imported_df['_temp_key'] = imported_df['Hog ID'].astype(str) + '_' + imported_df['Date'].astype(str)
                            existing_data_cleaned = st.session_state['hog_data'].dropna(subset=['Hog ID', 'Date', 'Weight (kg)']).copy()
                            existing_data_cleaned['_temp_key'] = existing_data_cleaned['Hog ID'].astype(str) + '_' + existing_data_cleaned['Date'].astype(str)
                            existing_data_filtered = existing_data_cleaned[
                                ~existing_data_cleaned['_temp_key'].isin(imported_df['_temp_key'])
                            ].drop(columns=['_temp_key'])
                            st.session_state['hog_data'] = existing_data_filtered
                            st.session_state['hog_data'] = pd.concat([st.session_state['hog_data'], imported_df.drop(columns=['_temp_key'])], ignore_index=True)
                            st.session_state['hog_data'].drop_duplicates(subset=['Hog ID', 'Date'], inplace=True, keep='last')
                            st.session_state['hog_data'].sort_values(by=['Hog ID', 'Date'], inplace=True)
                            save_data(st.session_state['hog_data'])
                            st.success("Data successfully imported from Excel!")
                        else:
                            st.error("Uploaded Excel file must contain 'Hog ID', 'Date', and 'Weight (kg)' columns, or be in the pivoted 'Hog ID' vs 'WEEK' (date columns) format.")
                    except Exception as e:
                        st.error(f"Error importing file: {e}")

    st.title("Hog Weight Tracking App")

    # Robust error handling for data loading
    try:
        if 'hog_data' not in st.session_state:
            st.session_state['hog_data'] = load_data()
    except Exception as e:
        st.session_state['hog_data'] = pd.DataFrame(columns=['Hog ID', 'Date', 'Weight (kg)'])
        st.error(f"Error loading hog data: {e}")

    try:
        if 'financial_data' not in st.session_state:
            st.session_state['financial_data'] = load_financial_data()
    except Exception as e:
        st.session_state['financial_data'] = pd.DataFrame(columns=['Date', 'Type', 'Category', 'Description', 'Hog ID', 'Weight (kg)', 'Price/kg', 'Amount'])
        st.error(f"Error loading financial data: {e}")

    # --- Make display_data available to all tabs and roles ---
    display_data = st.session_state['hog_data'].dropna(subset=['Hog ID', 'Date', 'Weight (kg)'])

    # --- Sidebar for Hog Management, Deletion, and Import ---
    # (Removed duplicate sidebar sections; all sidebar widgets are handled in the main sidebar block above)

    # Create tabs - show all tabs to everyone, but control content based on role
    tab_names = ["Data Entry & View", "Analysis", "Summary Report", "Financials"]
    if st.session_state.get('user_role') == 'admin':
        tab_names.append("Audit Trail")
    tabs = st.tabs(tab_names)
    tab1, tab2, tab3, tab4 = tabs[:4]
    tab5 = tabs[4] if len(tabs) > 4 else None
    # --- Audit Trail Setup ---
    import datetime
    if 'audit_trail' not in st.session_state:
        st.session_state['audit_trail'] = []  # List of dicts: {user, timestamp, action, record_type, record_id, field, old, new}

    def log_audit(action, record_type, record_id, field=None, old=None, new=None):
        st.session_state['audit_trail'].append({
            'user': st.session_state.get('username', 'unknown'),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'action': action,
            'record_type': record_type,
            'record_id': record_id,
            'field': field,
            'old': old,
            'new': new
        })

    with tab1:
        # Modern Header
        st.markdown("""
        <style>
            .data-entry-header {
                background: linear-gradient(135deg, #4b6cb7, #182848);
                color: white;
                padding: 1.5rem 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .metric-card {
                background: white;
                border-radius: 10px;
                padding: 1.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 1.5rem;
                border-left: 4px solid #4b6cb7;
            }
            .section-title {
                color: #2c3e50;
                border-bottom: 2px solid #eaeaea;
                padding-bottom: 0.5rem;
                margin-top: 1.5rem;
                font-weight: 600;
            }
            .dataframe {
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .dataframe th {
                background-color: #4b6cb7 !important;
                color: white !important;
                font-weight: 600 !important;
            }
            .dataframe td {
                padding: 8px 12px !important;
            }
            .stButton>button {
                width: 100%;
                border-radius: 6px;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
        </style>
        
        <div class='data-entry-header'>
            <h1 style='color: white; margin: 0;'>🐷 Hog Weight Management</h1>
            <p style='opacity: 0.9; margin: 0.5rem 0 0;'>Track and manage hog weight records with ease</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show view-only mode message for viewers
        if st.session_state['user_role'] == 'viewer':
            st.warning("🔒 You are in view-only mode. Viewers can see data but cannot make changes.", icon="⚠️")
        
        # Record Weight Measurement Section
        with st.container():
            st.markdown("### ⚖️ Record Weight Measurement")
            available_hogs = st.session_state['hog_data']['Hog ID'].unique()
            available_hogs_formatted = [f'{int(hog):03d}' for hog in available_hogs if pd.notna(hog)]

            import datetime
            if len(available_hogs_formatted) > 0:
                # Set/reset defaults for the form fields
                if 'record_hog_id' not in st.session_state or st.session_state['record_hog_id'] not in available_hogs_formatted:
                    st.session_state['record_hog_id'] = available_hogs_formatted[0]
                if 'record_date' not in st.session_state:
                    st.session_state['record_date'] = datetime.date.today()
                if 'record_weight' not in st.session_state:
                    st.session_state['record_weight'] = 0.0
                
                # Create a form container
                form_container = st.container()
                
                # Create form for better validation
                with form_container:
                    with st.form("weight_record_form"):
                        # Use responsive columns for better layout
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            selected_hog_id_formatted = st.selectbox(
                                "Select Hog ID:",
                                available_hogs_formatted,
                                key='record_hog_id',
                                index=available_hogs_formatted.index(st.session_state['record_hog_id']) if st.session_state['record_hog_id'] in available_hogs_formatted else 0,
                                help="Select the hog you want to record weight for"
                            )
                            selected_hog_id = int(selected_hog_id_formatted) if selected_hog_id_formatted else None
                        
                        with col2:
                            measurement_date = st.date_input(
                                "Measurement Date:",
                                key='record_date',
                                help="Select the date of measurement"
                            )
                        
                        with col3:
                            weight = st.number_input(
                                "Weight (kg):",
                                min_value=0.0,
                                step=0.1,
                                format="%.1f",
                                key='record_weight',
                                help="Enter the weight in kilograms"
                            )
                        
                        # Submit button with loading state
                        submit_button = st.form_submit_button("➕ Add Weight Record", type="primary")
                        
                        # Store form data in session state for confirmation
                        if submit_button:
                            if not selected_hog_id:
                                st.error("❌ Please select a Hog ID.")
                            elif not measurement_date:
                                st.error("❌ Please select a measurement date.")
                            elif weight is None or weight <= 0:
                                st.error("❌ Weight must be greater than 0.")
                            else:
                                # Store the form data for confirmation
                                st.session_state['pending_record'] = {
                                    'hog_id': selected_hog_id,
                                    'hog_id_formatted': selected_hog_id_formatted,
                                    'date': measurement_date,
                                    'weight': weight
                                }
                                st.rerun()
                
                # Show confirmation dialog if we have a pending record (outside the form container)
                if 'pending_record' in st.session_state and st.session_state['pending_record']:
                    pending = st.session_state['pending_record']
                    
                    st.warning("### 🔍 Confirm Weight Record")
                    st.markdown(f"""
                    - **Hog ID:** {pending['hog_id_formatted']}
                    - **Date:** {pending['date']}
                    - **Weight:** {pending['weight']:.1f} kg
                    """)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Confirm & Save", type="primary", key="confirm_save_btn"):
                            try:
                                new_record = pd.DataFrame([{
                                    'Hog ID': pending['hog_id'], 
                                    'Date': pending['date'], 
                                    'Weight (kg)': pending['weight']
                                }])
                                
                                # Update data
                                st.session_state['hog_data'] = pd.concat(
                                    [st.session_state['hog_data'], new_record], 
                                    ignore_index=True
                                )
                                st.session_state['hog_data'].drop_duplicates(
                                    subset=['Hog ID', 'Date'], 
                                    inplace=True, 
                                    keep='last'
                                )
                                st.session_state['hog_data'].sort_values(
                                    by=['Hog ID', 'Date'], 
                                    inplace=True
                                )
                                
                                # Save data
                                save_data(st.session_state['hog_data'])
                                st.success(f"✅ Successfully recorded weight of {pending['weight']:.1f} kg for Hog ID {pending['hog_id_formatted']} on {pending['date']}.")
                                
                                # Clear the pending record and force a rerun
                                del st.session_state['pending_record']
                                # Reset the form by rerunning without the record_weight in session state
                                if 'record_weight' in st.session_state:
                                    del st.session_state['record_weight']
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error saving record: {str(e)}")
                    
                    with col2:
                        if st.button("❌ Cancel", key="cancel_btn"):
                            # Clear the pending record
                            del st.session_state['pending_record']
                            st.rerun()
            else:
                st.warning("No hogs available. Please add hogs in the sidebar first.")
        
        # Hog Weight Data Section
        st.markdown("---")
        st.markdown("### 📋 Hog Weight Records")
        
        if not display_data.empty:
            # Add a search and filter section
            with st.expander("🔍 Search & Filter", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    search_hog = st.text_input("Search by Hog ID:", "", placeholder="e.g., 001, 002")
                with col2:
                    date_range = st.date_input(
                        "Filter by Date Range:",
                        value=[],
                        max_value=datetime.date.today(),
                        format="DD/MM/YYYY"
                    )
            
            # Apply filters
            filtered_display = display_data.copy()
            
            if search_hog:
                try:
                    search_id = int(search_hog)
                    filtered_display = filtered_display[filtered_display['Hog ID'] == search_id]
                except ValueError:
                    st.warning("Please enter a valid Hog ID number")
            
            if len(date_range) == 2:
                filtered_display = filtered_display[
                    (filtered_display['Date'] >= pd.Timestamp(date_range[0])) &
                    (filtered_display['Date'] <= pd.Timestamp(date_range[1]))
                ]
            
            # Create pivot table with Hog ID as rows and dates as columns
            if not filtered_display.empty:
                pivot_data = filtered_display.pivot_table(
                    index='Hog ID', 
                    columns='Date', 
                    values='Weight (kg)', 
                    aggfunc='last'  # Use last value if multiple entries for same hog/date
                )
                
                if not pivot_data.empty:
                    # Sort columns (dates) in descending order
                    pivot_data = pivot_data.reindex(sorted(pivot_data.columns, reverse=True), axis=1)
                    
                    # Format Hog IDs as three-digit integers and sort
                    pivot_data.index = pivot_data.index.astype(int)
                    pivot_data = pivot_data.sort_index()
                    
                    # Format index (Hog IDs) to show as three digits
                    pivot_data.index = [f'{int(hog_id):03d}' for hog_id in pivot_data.index]
                    
                    # Display the pivot table with enhanced styling
                    # Define a custom style function to handle None values
                    def style_weight(val):
                        if pd.isna(val):
                            return 'color: #999999; background-color: #f5f5f5'  # Light gray for None values
                        return ''
                    
                    st.dataframe(
                        pivot_data.style
                            .format(precision=1, na_rep='None')
                            .background_gradient(cmap='YlGnBu', axis=None, vmin=0)  # Ensure consistent color scale
                            .set_properties(**{
                                'text-align': 'center',
                                'border': '1px solid #e0e0e0',
                                'min-width': '100px'  # Ensure consistent column width
                            })
                            .applymap(style_weight),  # Apply custom style for None values
                        use_container_width=True,
                        height=min(600, 35 * (len(pivot_data) + 1))
                    )
                    
                    # Add export buttons in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CSV Export Button
                        st.download_button(
                            label="📥 Export to CSV",
                            data=pivot_data.to_csv().encode('utf-8'),
                            file_name=f'hog_weight_records_{datetime.datetime.now().strftime("%Y%m%d")}.csv',
                            mime='text/csv',
                            use_container_width=True,
                            help="Export data as CSV file"
                        )
                    
                    with col2:
                        # Excel Export Button
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            pivot_data.to_excel(writer, sheet_name='Hog Weight Records')
                            workbook = writer.book
                            worksheet = writer.sheets['Hog Weight Records']
                            
                            # Add some basic formatting
                            header_format = workbook.add_format({
                                'bold': True,
                                'text_wrap': True,
                                'valign': 'top',
                                'fg_color': '#4b6cb7',
                                'border': 1,
                                'color': 'white'
                            })
                            
                            # Format the header row
                            for col_num, value in enumerate(pivot_data.columns.values):
                                worksheet.write(0, col_num + 1, value, header_format)
                            
                            # Set column widths
                            for i, col in enumerate(pivot_data.columns):
                                max_length = max(\
                                    pivot_data[col].astype(str).apply(len).max(),
                                    len(str(col))
                                ) + 2  # Add a little extra space
                                worksheet.set_column(i + 1, i + 1, min(max_length, 15))
                        
                        excel_data = output.getvalue()
                        st.download_button(
                            label="📊 Export to Excel",
                            data=excel_data,
                            file_name=f'hog_weight_records_{datetime.datetime.now().strftime("%Y%m%d")}.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            use_container_width=True,
                            help="Export data as Excel file"
                        )
                else:
                    st.info("No records match the selected filters.")
            else:
                st.info("No weight records available for the selected filters.")
        else:
            st.info("No weight records available. Start by adding weight records using the form above.")
    # --- Analysis Tab ---
    with tab2:
        st.header("Analysis")
        # Show view-only mode message for viewers
        if st.session_state['user_role'] == 'viewer':
            st.info("🔍 You are in view-only mode. Viewers can see analysis data but cannot make changes.")

        if not display_data.empty:
            # Weight Trend Analysis
            st.subheader("Weight Trend Analysis")
            
            # Get unique hogs and sort them
            unique_hogs = sorted(display_data['Hog ID'].unique())
            formatted_hogs = [f"{int(hog):03d}" for hog in unique_hogs]
            hog_dict = dict(zip(formatted_hogs, unique_hogs))
            
            # Initialize session state for hog selection if it doesn't exist
            if 'hog_selection' not in st.session_state:
                st.session_state.hog_selection = formatted_hogs[:min(3, len(formatted_hogs))]
            
            # Handle select all functionality
            if st.button("Select All Hogs"):
                st.session_state.hog_selection = formatted_hogs.copy()
                st.rerun()
            
            # Create columns for better layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Allow selection of multiple hogs for comparison
                selected_hogs_formatted = st.multiselect(
                    "Select Hogs to Compare:",
                    options=formatted_hogs,
                    default=st.session_state.hog_selection,
                    format_func=lambda x: x,
                    key='hog_selector'
                )
                
                # Update session state when selection changes
                if selected_hogs_formatted != st.session_state.hog_selection:
                    st.session_state.hog_selection = selected_hogs_formatted.copy()
                    st.rerun()
            
            # Convert formatted hog IDs back to original values for filtering
            selected_hogs = [hog_dict[hog] for hog in st.session_state.hog_selection if hog in hog_dict]
            
            if selected_hogs:
                # Filter data for selected hogs
                filtered_data = display_data[display_data['Hog ID'].isin(selected_hogs)]
                
                # Create line chart with Plotly
                fig = px.line(
                    filtered_data, 
                    x='Date', 
                    y='Weight (kg)',
                    color='Hog ID',
                    title='Weight Trend by Hog ID',
                    labels={'Date': 'Date', 'Weight (kg)': 'Weight (kg)', 'Hog ID': 'Hog ID'},
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                
                # Update layout for better readability
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Weight (kg)',
                    legend_title='Hog ID',
                    hovermode='x unified',
                    height=500
                )
                
                # Modern Header
                st.markdown("""
                <style>
                    .main-header {
                        background: linear-gradient(135deg, #4b6cb7, #182848);
                        color: white;
                        padding: 1.5rem 2rem;
                        border-radius: 10px;
                        margin-bottom: 2rem;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }
                    .metric-card {
                        background: white;
                        border-radius: 10px;
                        padding: 1rem;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        border-left: 4px solid #4b6cb7;
                        margin-bottom: 1rem;
                    }
                    .metric-value {
                        font-size: 1.5rem;
                        font-weight: 600;
                        color: #2c3e50;
                    }
                    .metric-label {
                        font-size: 0.9rem;
                        color: #7f8c8d;
                        margin-bottom: 0.5rem;
                    }
                    .section-title {
                        color: #2c3e50;
                        border-bottom: 2px solid #eaeaea;
                        padding-bottom: 0.5rem;
                        margin-top: 1.5rem;
                    }
                    .dataframe {
                        border-radius: 8px;
                        overflow: hidden;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .dataframe th {
                        background-color: #4b6cb7 !important;
                        color: white !important;
                        font-weight: 600 !important;
                    }
                    .dataframe td {
                        padding: 8px 12px !important;
                    }
                </style>
                
                <div class='main-header'>
                    <h1 style='color: white; margin: 0;'>🐷 Hog Performance Analytics</h1>
                    <p style='opacity: 0.9; margin: 0.5rem 0 0;'>Comprehensive analysis of hog growth and performance metrics</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Weight Trend Section
                with st.container():
                    st.markdown("### 📈 Weight Trend Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add summary metrics
                    st.markdown("### 📊 Performance Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class='metric-card' style='border-left-color: #4b6cb7;'>
                            <div class='metric-label'>Total Hogs Tracked</div>
                            <div class='metric-value'>{len(selected_hogs) if selected_hogs else '0'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        avg_weight = filtered_data['Weight (kg)'].mean()
                        weight_value = f"{avg_weight:.1f} kg" if not pd.isna(avg_weight) else "-"
                        st.markdown(f"""
                        <div class='metric-card' style='border-left-color: #2ecc71;'>
                            <div class='metric-label'>Average Weight</div>
                            <div class='metric-value'>{weight_value}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col3:
                        total_records = len(filtered_data)
                        st.markdown(f"""
                        <div class='metric-card' style='border-left-color: #e74c3c;'>
                            <div class='metric-label'>Total Records</div>
                            <div class='metric-value'>{total_records}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col4:
                        days_tracked = (filtered_data['Date'].max() - filtered_data['Date'].min()).days if len(filtered_data) > 1 else 0
                        days_display = days_tracked if days_tracked > 0 else "-"
                        st.markdown(f"""
                        <div class='metric-card' style='border-left-color: #9b59b6;'>
                            <div class='metric-label'>Days Tracked</div>
                            <div class='metric-value'>{days_display}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Summary Statistics Section
                with st.expander("📋 View Detailed Statistics", expanded=False):
                    st.markdown("### 🐖 Individual Hog Statistics")
                    stats = filtered_data.groupby('Hog ID')['Weight (kg)'].agg(['min', 'max', 'mean', 'count']).reset_index()
                    stats.columns = ['Hog ID', 'Min Weight (kg)', 'Max Weight (kg)', 'Avg Weight (kg)', 'Records']
                    stats['Hog ID'] = stats['Hog ID'].apply(lambda x: f"{int(x):03d}")
                    
                    st.dataframe(
                        stats.style\
                            .background_gradient(subset=['Avg Weight (kg)'], cmap='YlGnBu')\
                            .set_properties(**{'text-align': 'left', 'border': '1px solid #e0e0e0'})\
                            .format({
                                'Min Weight (kg)': '{:.1f}',
                                'Max Weight (kg)': '{:.1f}',
                                'Avg Weight (kg)': '{:.1f}'
                            }),
                        hide_index=True, 
                        use_container_width=True
                    )
                
                # Growth Analysis Section
                st.markdown("### 📈 Growth Analysis")
                growth_rates = []
                for hog_id in selected_hogs:
                    hog_data = filtered_data[filtered_data['Hog ID'] == hog_id].sort_values('Date')
                    if len(hog_data) > 1:
                        initial_weight = hog_data['Weight (kg)'].iloc[0]
                        final_weight = hog_data['Weight (kg)'].iloc[-1]
                        days = (hog_data['Date'].iloc[-1] - hog_data['Date'].iloc[0]).days
                        adg = (final_weight - initial_weight) / days if days > 0 else 0
                        growth_rates.append({
                            'Hog ID': f"{int(hog_id):03d}",
                            'Initial Weight (kg)': f"{initial_weight:.1f}",
                            'Final Weight (kg)': f"{final_weight:.1f}",
                            'Total Gain (kg)': f"{final_weight - initial_weight:.1f}",
                            'Days Tracked': days,
                            'ADG (kg/day)': f"{adg:.3f}"
                        })
                
                if not selected_hogs:
                    st.warning("⚠️ Please select at least one hog to view growth analysis.")
                elif growth_rates:
                    growth_df = pd.DataFrame(growth_rates)
                    
                    # Convert string values back to float for calculations
                    for col in ['Initial Weight (kg)', 'Final Weight (kg)', 'Total Gain (kg)', 'ADG (kg/day)']:
                        growth_df[col] = growth_df[col].astype(float)
                    
                    # Add growth percentage
                    growth_df['Growth %'] = (growth_df['Total Gain (kg)'] / growth_df['Initial Weight (kg)']) * 100
                    
                    # Growth Summary Cards
                    st.markdown("#### 📊 Growth Summary")
                    g_col1, g_col2, g_col3 = st.columns(3)
                    
                    with g_col1:
                        avg_adg = growth_df['ADG (kg/day)'].mean()
                        st.markdown(f"""
                        <div class='metric-card' style='background: linear-gradient(135deg, #36D1DC, #5B86E5); color: white;'>
                            <div class='metric-label'>Avg. Daily Gain</div>
                            <div class='metric-value' style='color: white;'>{avg_adg:.3f} kg/day</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with g_col2:
                        total_gain = growth_df['Total Gain (kg)'].sum()
                        st.markdown(f"""
                        <div class='metric-card' style='background: linear-gradient(135deg, #11998e, #38ef7d); color: white;'>
                            <div class='metric-label'>Total Weight Gain</div>
                            <div class='metric-value' style='color: white;'>{total_gain:.1f} kg</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with g_col3:
                        avg_growth_pct = growth_df['Growth %'].mean()
                        st.markdown(f"""
                        <div class='metric-card' style='background: linear-gradient(135deg, #8E2DE2, #4A00E0); color: white;'>
                            <div class='metric-label'>Avg. Growth</div>
                            <div class='metric-value' style='color: white;'>{avg_growth_pct:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display growth metrics in a styled table
                    st.markdown("#### 📝 Individual Growth Metrics")
                    st.dataframe(
                        growth_df.style\
                            .format({
                                'Initial Weight (kg)': '{:.1f}',
                                'Final Weight (kg)': '{:.1f}',
                                'Total Gain (kg)': '{:.1f}',
                                'ADG (kg/day)': '{:.3f}',
                                'Growth %': '{:.1f}%'
                            })\
                            .background_gradient(subset=['Total Gain (kg)', 'ADG (kg/day)', 'Growth %'], cmap='YlGn')\
                            .set_properties(subset=['Hog ID'], **{'font-weight': 'bold'}),
                        hide_index=True, 
                        use_container_width=True
                    )
                    
                    # Add insights section
                    st.markdown("#### 💡 Key Insights")
                    best_growth = growth_df.loc[growth_df['Total Gain (kg)'].idxmax()]
                    best_adg = growth_df.loc[growth_df['ADG (kg/day)'].idxmax()]
                    
                    insight_col1, insight_col2 = st.columns(2)
                    
                    with insight_col1:
                        with st.container(border=True):
                            st.markdown("##### 🏆 Top Performer")
                            st.markdown(f"""
                            **Hog {best_growth['Hog ID']}** achieved the highest total weight gain:
                            - **Gained:** {best_growth['Total Gain (kg)']:.1f} kg
                            - **Growth:** {best_growth['Growth %']:.1f}%
                            - **Duration:** {best_growth['Days Tracked']} days
                            - **ADG:** {best_growth['ADG (kg/day)']:.3f} kg/day
                            """)
                    
                    with insight_col2:
                        with st.container(border=True):
                            st.markdown("##### ⚡ Fastest Gainer")
                            st.markdown(f"""
                            **Hog {best_adg['Hog ID']}** had the highest daily gain:
                            - **ADG:** {best_adg['ADG (kg/day)']:.3f} kg/day
                            - **Total Gain:** {best_adg['Total Gain (kg)']:.1f} kg
                            - **Duration:** {best_adg['Days Tracked']} days
                            - **Growth:** {best_adg['Growth %']:.1f}%
                            """)
                else:
                    st.info("ℹ️ Not enough data to calculate growth rates. Need at least two weight records per hog.")
            
            # Week-to-Week Growth Section
            st.markdown("---")
            st.markdown("### 📅 Week-to-Week Growth Per Hog")
            
            # Calculate weekly growth metrics for each hog
            weekly_growth_data = []
            hog_summary = []
            
            # Make sure we have a copy of the data to work with
            working_data = display_data.copy()
            
            # Convert Date to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(working_data['Date']):
                working_data['Date'] = pd.to_datetime(working_data['Date'])
            
            for hog_id in working_data['Hog ID'].unique():
                hog_data = working_data[working_data['Hog ID'] == hog_id].sort_values('Date')
                if len(hog_data) > 1:
                    # Calculate weekly growth
                    hog_data = hog_data.copy()
                    hog_data['Week'] = hog_data['Date'].dt.strftime('%Y-W%U')
                    weekly_data = hog_data.groupby('Week')['Weight (kg)'].mean().reset_index()
                    
                    # Calculate growth between weeks
                    weekly_data['Previous Weight'] = weekly_data['Weight (kg)'].shift(1)
                    weekly_data = weekly_data.dropna()
                    
                    if not weekly_data.empty:
                        weekly_data['Weekly Growth (kg)'] = weekly_data['Weight (kg)'] - weekly_data['Previous Weight']
                        weekly_data['Weekly Growth (%)'] = (weekly_data['Weekly Growth (kg)'] / weekly_data['Previous Weight']) * 100
                        
                        # Add to weekly growth data
                        for _, row in weekly_data.iterrows():
                            weekly_growth_data.append({
                                'Hog ID': hog_id,
                                'Week': row['Week'],
                                'Weight (kg)': row['Weight (kg)'],
                                'Weekly Growth (kg)': row['Weekly Growth (kg)'],
                                'Weekly Growth (%)': row['Weekly Growth (%)']
                            })
                        
                        # Calculate summary for this hog
                        total_growth_pct = ((weekly_data['Weight (kg)'].iloc[-1] - weekly_data['Weight (kg)'].iloc[0]) / 
                                         weekly_data['Weight (kg)'].iloc[0]) * 100
                        avg_weekly_growth = weekly_data['Weekly Growth (%)'].mean()
                        
                        hog_summary.append({
                            'Hog ID': hog_id,
                            'Start Weight (kg)': weekly_data['Weight (kg)'].iloc[0],
                            'End Weight (kg)': weekly_data['Weight (kg)'].iloc[-1],
                            'Total Growth (kg)': weekly_data['Weight (kg)'].iloc[-1] - weekly_data['Weight (kg)'].iloc[0],
                            'Total Growth (%)': total_growth_pct,
                            'Avg Weekly Growth (%)': avg_weekly_growth,
                            'Weeks Tracked': len(weekly_data)
                        })
            
            if weekly_growth_data and hog_summary:
                # Create summary metrics
                summary_df = pd.DataFrame(hog_summary)
                weekly_df = pd.DataFrame(weekly_growth_data)
                
                # Calculate overall metrics
                overall_avg_weekly_growth = weekly_df['Weekly Growth (%)'].mean()
                positive_weeks = len(weekly_df[weekly_df['Weekly Growth (%)'] > 0])
                negative_weeks = len(weekly_df[weekly_df['Weekly Growth (%)'] < 0])
                
                # Display summary metrics
                st.markdown("### Growth Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Avg Weekly Growth", f"{overall_avg_weekly_growth:.1f}%")
                with col2:
                    st.metric("Weeks with Growth", f"{positive_weeks}")
                with col3:
                    st.metric("Weeks with Decline", f"{negative_weeks}")
                
                # Display weekly growth in a cleaner format
                st.markdown("### Weekly Growth")
                
                # Create a pivot table with weeks in reverse chronological order
                weekly_df['Date'] = pd.to_datetime(weekly_df['Week'] + '-1', format='%Y-W%U-%w')
                weekly_df = weekly_df.sort_values('Date', ascending=False)
                
                # Create pivot table with sorted weeks
                pivot_df = weekly_df.pivot(
                    index='Hog ID',
                    columns='Week',
                    values='Weekly Growth (%)'
                ).round(1)
                
                # Ensure we have the latest week's data first
                if not pivot_df.empty:
                    # Get the most recent week's growth for each hog
                    latest_week = weekly_df['Week'].iloc[0]  # Already sorted by date
                    latest_growth = weekly_df[weekly_df['Week'] == latest_week].set_index('Hog ID')['Weekly Growth (%)']
                    
                    # Add latest growth as the first column
                    pivot_df.insert(0, 'Latest Growth', latest_growth)
                    
                    # Sort columns to ensure Latest Growth is first, then sort weeks in reverse chronological order
                    week_columns = sorted([col for col in pivot_df.columns if col != 'Latest Growth'], 
                                        key=lambda x: pd.to_datetime(x + '-1', format='%Y-W%U-%w'), 
                                        reverse=True)
                    pivot_df = pivot_df[['Latest Growth'] + week_columns]
                    
                    # Style the table with color coding
                    def color_growth(val):
                        if pd.isna(val):
                            return ''
                        color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                        return f'color: {color}; font-weight: bold;'
                    
                    # Display the table with expandable details
                    with st.expander("📈 View Weekly Growth Details", expanded=False):
                        st.dataframe(
                            pivot_df.style.applymap(color_growth)
                            .format('{:.1f}%', na_rep='-')
                            .set_properties(**{'text-align': 'center'}),
                            use_container_width=True
                        )
                
                # Outlier Detection
                with st.expander("🔍 Outlier Detection", expanded=False):
                    st.markdown("### 🚩 Potential Data Issues")
                    
                    # Explanation of what outliers are
                    with st.expander("ℹ️ What are potential data issues?", expanded=True):
                        st.markdown("""
                        Potential data issues, or 'outliers', are weight measurements that are unusually different from others. 
                        These could be caused by:
                        - 🖊️ Typing mistakes when recording weights
                        - ⚖️ Scale errors or miscalibrations
                        - 🐖 Unusual growth patterns that might need attention
                        - 📅 Incorrect date entries
                        
                        We flag measurements that are more than 3 standard deviations from the average as potential issues.
                        """)
                    
                    if not display_data.empty:
                        # Create a copy of the data for analysis
                        analysis_data = display_data.copy()
                        
                        # Calculate Z-scores for weight measurements using pandas' abs() method
                        analysis_data['z_score'] = ((analysis_data['Weight (kg)'] - analysis_data['Weight (kg)'].mean()) / analysis_data['Weight (kg)'].std()).abs()
                        
                        # Identify outliers (z-score > 3)
                        outliers = analysis_data[analysis_data['z_score'] > 3].sort_values('z_score', ascending=False)
                        
                        if not outliers.empty:
                            st.warning(f"⚠️ Found {len(outliers)} unusual weight measurements that need review")
                            
                            # Explanation of what to do with outliers
                            st.info("""
                            **What to do with these results?**
                            - Check if these weights were recorded correctly
                            - Verify the scale was working properly on these dates
                            - Look for patterns (e.g., same hog appearing multiple times might indicate a growth issue)
                            - Consider if these weights are realistic for the hogs' age and breed
                            """)
                            
                            # Show outliers in a table
                            st.markdown("### 📋 Unusual Measurements")
                            st.dataframe(
                                outliers[['Hog ID', 'Date', 'Weight (kg)', 'z_score']]
                                .sort_values('z_score', ascending=False)
                                .style.format({
                                    'Weight (kg)': '{:.1f}',
                                    'z_score': '{:.2f}'
                                })
                                .apply(lambda x: ['background-color: #fff3cd' if x.name in outliers.index else '' for i in x], axis=1)
                                .set_properties(**{'border': '1px solid #ffcccb'}),
                                use_container_width=True
                            )
                            
                            # Show distribution of weights
                            st.markdown("### 📊 Weight Distribution")
                            fig = px.histogram(
                                display_data, 
                                x='Weight (kg)',
                                title='Distribution of Hog Weights',
                                marginal='box',
                                nbins=30
                            )
                            
                            # Add vertical lines for mean and standard deviations
                            mean_weight = display_data['Weight (kg)'].mean()
                            std_weight = display_data['Weight (kg)'].std()
                            
                            for i in range(1, 4):
                                fig.add_vline(x=mean_weight + (i * std_weight), line_dash='dash', line_color='red', 
                                           annotation_text=f'+{i}σ', annotation_position='top right')
                                fig.add_vline(x=mean_weight - (i * std_weight), line_dash='dash', line_color='red', 
                                           annotation_text=f'-{i}σ', annotation_position='top right')
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Explanation of the distribution chart
                            st.markdown("""
                            **Understanding the chart:**
                            - The bars show how many hogs fall into each weight range
                            - The box plot on top shows the spread of weights
                            - Red dashed lines show standard deviations from the mean
                            - Measurements outside the ±3σ range are considered unusual
                            """)
                            
                        else:
                            st.success("✅ No unusual weight measurements detected. All weights appear to be within expected ranges.")
                            st.info("""
                            This is great news! It means:
                            - Your weight recordings are consistent
                            - The scale appears to be working correctly
                            - Hogs are growing at expected rates
                            - No immediate data entry issues detected
                            """)
                    else:
                        st.info("No data available for outlier detection.")
                
                # Weekly Performance Dashboard (Collapsible)
                with st.expander("🏆 Weekly Performance Dashboard", expanded=False):
                    st.markdown("### Weekly Performance Overview")
                    
                    if not weekly_df.empty:
                        # Get unique weeks and sort them in descending order (newest first)
                        available_weeks = sorted(weekly_df['Week'].unique(), 
                                              key=lambda x: pd.to_datetime(x + '-1', format='%Y-W%U-%w'), 
                                              reverse=True)
                        
                        # Display current week info
                        current_week = available_weeks[0] if available_weeks else None
                        
                        # Add week selector dropdown with current week highlighted
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            selected_week = st.selectbox(
                                "Select Week to View:",
                                available_weeks,
                                index=0,  # Default to most recent week
                                format_func=lambda x: f"📅 Week {x}" + (" (Current)" if x == current_week else "")
                            )
                        
                        # Show current week's summary
                        if selected_week == current_week:
                            st.success(f"🔍 Showing latest data for Week {current_week}")
                        else:
                            st.info(f"ℹ️ Viewing historical data for Week {selected_week}")
                        
                        # Get data for selected week
                        week_data = weekly_df[weekly_df['Week'] == selected_week]
                    else:
                        st.info("No weekly performance data available.")
                        return
                    
                    if not week_data.empty:
                        # Get top and bottom performers for the selected week
                        top_weekly = week_data.nlargest(3, 'Weekly Growth (%)')
                        bottom_weekly = week_data.nsmallest(3, 'Weekly Growth (%)')
                        
                        # Display weekly top performers
                        st.markdown(f"### 🏆 Top Performers (Week {selected_week})")
                        if len(top_weekly) > 0:
                            cols = st.columns(3)
                            for i, (_, row) in enumerate(top_weekly.iterrows()):
                                with cols[i % 3]:  # Use modulo to handle cases with < 3 hogs
                                    st.metric(
                                        f"🥇 #{i+1} (Hog {row['Hog ID']})",
                                        f"+{row['Weekly Growth (%)']:.1f}%",
                                        f"{row['Weekly Growth (kg)']:+.1f} kg"
                                    )
                        else:
                            st.info("No positive growth data for this week.")
                        
                        # Display weekly bottom performers
                        st.markdown(f"### 📉 Needs Attention (Week {selected_week})")
                        if len(bottom_weekly) > 0:
                            cols = st.columns(3)
                            for i, (_, row) in enumerate(bottom_weekly.iterrows()):
                                with cols[i % 3]:  # Use modulo to handle cases with < 3 hogs
                                    st.metric(
                                        f"🔻 #{i+1} (Hog {row['Hog ID']})",
                                        f"{row['Weekly Growth (%)']:+.1f}%",
                                        f"{row['Weekly Growth (kg)']:+.1f} kg"
                                    )
                        else:
                            st.info("No negative growth data for this week.")
                
                # Display full hog summary in a collapsible section
                with st.expander("📊 View All Hogs Summary", expanded=False):
                    st.dataframe(
                        summary_df.sort_values('Total Growth (%)', ascending=False)
                        .style.format({
                            'Start Weight (kg)': '{:.1f}',
                            'End Weight (kg)': '{:.1f}',
                            'Total Growth (kg)': '{:.1f}',
                            'Total Growth (%)': '{:.1f}%',
                            'Avg Weekly Growth (%)': '{:.1f}%'
                        }),
                        hide_index=True,
                        use_container_width=True
                    )
                
            else:
                st.info("Not enough weekly data to calculate growth metrics. Need at least two weeks of data.")
            
            # Always show the Average Weight Per Week section
            with st.expander("📊 Show/Hide Average Weight Per Week (All Hogs)", expanded=False):
                # Prepare data for weekly averages
                weekly_avg = display_data.copy()
                weekly_avg['Date'] = pd.to_datetime(weekly_avg['Date'])
                weekly_avg['Week'] = weekly_avg['Date'].dt.strftime('%Y-W%U')
                weekly_avg = weekly_avg.groupby('Week')['Weight (kg)'].mean().reset_index()
                
                if not weekly_avg.empty:
                    # Create line chart for weekly averages
                    fig_weekly = px.line(
                        weekly_avg,
                        x='Week',
                        y='Weight (kg)',
                        title='Average Weight Per Week (All Hogs)',
                        markers=True,
                        labels={'Weight (kg)': 'Average Weight (kg)', 'Week': 'Week'}
                    )
                    
                    # Update layout for better readability
                    fig_weekly.update_layout(
                        xaxis_title='Week',
                        yaxis_title='Average Weight (kg)',
                        hovermode='x unified',
                        height=500,
                        xaxis=dict(tickangle=45)
                    )
                    
                    # Display the plot
                    st.plotly_chart(fig_weekly, use_container_width=True)
                    
                    # Display the data table
                    st.dataframe(
                        weekly_avg.rename(columns={
                            'Week': 'Week',
                            'Weight (kg)': 'Avg Weight (kg)'
                        }).style.format({
                            'Avg Weight (kg)': '{:.2f}'
                        }),
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("Not enough data to calculate weekly averages.")
        else:
            st.info("No weight records available for analysis.")

    # --- Audit Trail Tab ---
    if tab5:
        with tab5:
            st.header("Audit Trail (Change History)")
            audit_df = pd.DataFrame(st.session_state['audit_trail'])
            if not audit_df.empty:
                st.dataframe(audit_df, use_container_width=True)
            else:
                st.info("No changes have been logged yet.")

            st.header("Hog Weight Data")
            display_data = st.session_state['hog_data'].dropna(subset=['Hog ID', 'Date', 'Weight (kg)'])
            if not display_data.empty:
                pivot_data = display_data.copy()
                pivot_data['Date'] = pd.to_datetime(pivot_data['Date'])
                pivot_data_display = pivot_data.pivot_table(index='Hog ID', columns='Date', values='Weight (kg)', aggfunc='first')
                pivot_data_display = pivot_data_display.iloc[:, ::-1]
                pivot_data_display.columns = pivot_data_display.columns.strftime('%d/%m/%Y')
                pivot_data_display.columns.name = 'WEEK'
                pivot_data_display.index = pivot_data_display.index.map(lambda x: f'{int(x):03d}')
                pivot_data_display.index.name = 'Hog ID'
                # Use container width for mobile
                st.dataframe(pivot_data_display, use_container_width=True, height=400)
                # Export data
                st.subheader("Export Data")
                col_export1, col_export2 = st.columns([1, 1])
                with col_export1:
                    csv_file = display_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Export as CSV",
                        data=csv_file,
                        file_name='hog_weight_data.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                with col_export2:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        pivot_sheet_name = 'Pivoted Weight Data'
                        raw_sheet_name = 'Raw Hog Data'
                        pivot_data_display.to_excel(writer, sheet_name=pivot_sheet_name, startrow=2)
                        worksheet_pivot = writer.sheets[pivot_sheet_name]
                        title_format = writer.book.add_format({'bold': True, 'font_size': 14})
                        worksheet_pivot.write('A1', 'Hog Weight Data - Weekly Overview', title_format)
                        display_data.to_excel(writer, index=False, sheet_name=raw_sheet_name)
                    excel_file_bytes = buffer.getvalue()
                    st.download_button(
                        label="Export as Excel",
                        data=excel_file_bytes,
                        file_name='hog_weight_data.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )
            else:
                st.info("No complete weight records to display or plot yet.")

    with tab3:
        # Custom CSS for modern, professional styling
        st.markdown("""
        <style>
            .metric-card {
                background: white;
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                border: 1px solid #e9ecef;
                transition: all 0.2s ease;
            }
            .metric-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                transform: translateY(-1px);
            }
            .metric-card h3 {
                color: #6c757d;
                font-size: 0.9rem;
                font-weight: 500;
                margin: 0 0 0.5rem 0;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .metric-card h2 {
                color: #2c3e50;
                font-size: 1.5rem;
                font-weight: 600;
                margin: 0;
            }
            .metric-card p {
                color: #6c757d;
                font-size: 0.85rem;
                margin: 0.5rem 0 0 0;
            }
            .section-header {
                color: #2c3e50;
                font-size: 1.25rem;
                font-weight: 600;
                margin: 1.5rem 0 1rem 0;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #f0f2f6;
            }
            .info-box {
                background: #f8f9fa;
                border-left: 3px solid #3498db;
                padding: 0.75rem 1rem;
                margin: 1rem 0;
                border-radius: 4px;
                font-size: 0.9rem;
                line-height: 1.5;
            }
            .highlight-box {
                background: #f8f9fa;
                border-radius: 6px;
                padding: 0.75rem 1rem;
                margin: 1rem 0;
                border: 1px solid #e9ecef;
                font-size: 0.9rem;
            }
        </style>
        """, unsafe_allow_html=True)

        # Page Header
        st.markdown("<h1 style='color: #2c3e50; margin-bottom: 1.5rem;'>📊 Farm Analytics Dashboard</h1>", unsafe_allow_html=True)
        
        # Show view-only mode message for viewers
        if st.session_state['user_role'] == 'viewer':
            st.markdown("""
            <div class='info-box'>
                <b>🔍 View-Only Mode</b><br>
                You are currently in view-only mode. You can view all summary reports but cannot make any changes.
            </div>
            """, unsafe_allow_html=True)

        # --- Average Weight Per Week Section ---
        with st.expander("📈 Average Weight Per Week (All Hogs)", expanded=True):
            if not display_data.empty:
                display_data_copy = display_data.copy()
                display_data_copy['Date'] = pd.to_datetime(display_data_copy['Date'])
                display_data_copy['Week'] = display_data_copy['Date'].dt.isocalendar().week.astype(int)
                display_data_copy['Year'] = display_data_copy['Date'].dt.year
                
                avg_weight_per_week = display_data_copy.groupby(['Year', 'Week'])['Weight (kg)'].mean().reset_index()
                avg_weight_per_week['YearWeek'] = avg_weight_per_week['Year'].astype(str) + "-W" + avg_weight_per_week['Week'].astype(str).str.zfill(2)
                avg_weight_per_week = avg_weight_per_week.sort_values(by='YearWeek')
                
                if not avg_weight_per_week.empty:
                    # Create a two-column layout for metrics and chart
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Calculate metrics
                        latest_week = avg_weight_per_week.iloc[-1]
                        prev_week = avg_weight_per_week.iloc[-2] if len(avg_weight_per_week) > 1 else None
                        
                        # Convert Series to scalar values for comparison
                        latest_weight = latest_week['Weight (kg)'].iloc[0] if isinstance(latest_week['Weight (kg)'], pd.Series) else latest_week['Weight (kg)']
                        
                        if prev_week is not None:
                            prev_weight = prev_week['Weight (kg)'].iloc[0] if isinstance(prev_week['Weight (kg)'], pd.Series) else prev_week['Weight (kg)']
                            weight_change = f"<p>▲ {((latest_weight - prev_weight) / prev_weight * 100):.1f}% from last week</p>"
                        else:
                            weight_change = ""
                            
                        st.markdown("<div class='metric-card'>" + 
                                  f"<h3>Current Average Weight</h3>" +
                                  f"<h2>{latest_weight:.1f} kg</h2>" +
                                  weight_change +
                                  "</div>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='metric-card'>" +
                                  f"<h3>Total Weeks Tracked</h3>" +
                                  f"<h2>{len(avg_weight_per_week)}</h2>" +
                                  (f"<p>Since {avg_weight_per_week['YearWeek'].iloc[0]}</p>" if len(avg_weight_per_week) > 0 else "") +
                                  "</div>", unsafe_allow_html=True)
                    
                    with col2:
                        # Enhanced chart with trendline
                        fig_avg_plotly = px.line(
                            avg_weight_per_week,
                            x='YearWeek',
                            y='Weight (kg)',
                            title="Average Weight Trend Over Time",
                            markers=True,
                            line_shape='spline',
                            template='plotly_white'
                        )
                        
                        # Add trendline
                        z = np.polyfit(range(len(avg_weight_per_week)), avg_weight_per_week['Weight (kg)'], 1)
                        p = np.poly1d(z)
                        fig_avg_plotly.add_scatter(
                            x=avg_weight_per_week['YearWeek'],
                            y=p(range(len(avg_weight_per_week))),
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='Trend Line'
                        )
                        
                        fig_avg_plotly.update_traces(
                            hovertemplate='<b>Week %{x}</b><br>Avg Weight: %{y:.1f} kg<extra></extra>',
                            line=dict(width=3)
                        )
                        
                        fig_avg_plotly.update_layout(
                            xaxis_title='Week',
                            yaxis_title='Average Weight (kg)',
                            hovermode='x unified',
                            showlegend=True,
                            height=400
                        )
                        
                        st.plotly_chart(fig_avg_plotly, use_container_width=True)
                        
                        # Display the weekly averages with a select box
                        st.markdown("<h4>📋 Weekly Averages</h4>", unsafe_allow_html=True)
                        
                        # Create a clean DataFrame for display
                        weekly_avg_df = pd.DataFrame({
                            'Week': avg_weight_per_week['YearWeek'],
                            'Avg Weight (kg)': avg_weight_per_week['Weight (kg)']
                        })
                        
                        # Add a select box to choose a specific week
                        selected_week = st.selectbox(
                            'Select a week to view details:',
                            options=weekly_avg_df['Week'].tolist(),
                            index=len(weekly_avg_df) - 1  # Default to the most recent week
                        )
                        
                        # Display the selected week's data
                        selected_data = weekly_avg_df[weekly_avg_df['Week'] == selected_week].iloc[0]
                        
                        # Show the selected week's data in a nice format
                        st.markdown(
                            f"<div class='metric-card' style='margin-top: 1rem;'>"
                            f"<h3>Selected Week: {selected_data['Week']}</h3>"
                            f"<h2 style='color: #2c3e50;'>{selected_data['Avg Weight (kg)']:.1f} kg</h2>"
                            "</div>",
                            unsafe_allow_html=True
                        )
                        
                        # Optional: Show a small table with the last 4 weeks for context
                        recent_weeks = weekly_avg_df.tail(4).sort_values('Week', ascending=False)
                        st.markdown("<p style='margin-top: 1rem;'><small>Recent weeks:</small></p>", unsafe_allow_html=True)
                        st.dataframe(
                            recent_weeks.style.format({
                                'Avg Weight (kg)': '{:.1f} kg'
                            }),
                            use_container_width=True,
                            hide_index=True,
                            height=min(180, 35 * len(recent_weeks) + 30)
                        )
                else:
                    st.info("No weekly average data available.")
            else:
                st.info("No weight data available for analysis.")

        # --- Advanced Analytics Section ---
        st.markdown("<h2 class='section-header'>📊 Advanced Analytics</h2>", unsafe_allow_html=True)
        
        # Create tabs for different analytics sections
        tab_insights, tab_finance, tab_efficiency, tab_growth = st.tabs([
            "🔍 Insights", "💰 Financials", "📈 Efficiency", "📊 Growth"
        ])
        
        with tab_insights:
            st.markdown("### 📊 Financial Insights")
            if not st.session_state['financial_data'].empty:
                fin_df = st.session_state['financial_data'].copy()
                fin_df['Date'] = pd.to_datetime(fin_df['Date'], errors='coerce')
                fin_df = fin_df.dropna(subset=['Date'])
                fin_df['Month'] = fin_df['Date'].dt.to_period('M')
                fin_df['Year'] = fin_df['Date'].dt.year
                
                # Calculate key metrics
                total_sales = fin_df[fin_df['Type'] == 'Sale']['Amount'].sum()
                total_expenses = fin_df[fin_df['Type'] == 'Expense']['Amount'].sum()
                net_profit = total_sales - total_expenses
                gross_profit_margin = ((total_sales - total_expenses) / total_sales * 100) if total_sales > 0 else 0
                operating_expense_ratio = (total_expenses / total_sales * 100) if total_sales > 0 else 0
                
                # Create tabs for different financial sections
                tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Trends", "📋 Details"])
                
                with tab1:  # Overview Tab
                    st.markdown("### 📊 Financial Overview")
                    
                    # Top Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Revenue", f"Kshs {total_sales:,.2f}")
                    with col2:
                        st.metric("Total Expenses", f"Kshs {total_expenses:,.2f}")
                    with col3:
                        st.metric("Net Profit", f"Kshs {net_profit:,.2f}", 
                                f"{'+' if net_profit >= 0 else ''}{(net_profit/total_sales*100) if total_sales > 0 else 0:.1f}%")
                    with col4:
                        st.metric("Profit Margin", f"{gross_profit_margin:.1f}%")
                    
                    # Charts Row 1
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Expense Breakdown
                        st.markdown("#### 💰 Expense Breakdown")
                        expense_by_category = fin_df[fin_df['Type'] == 'Expense'].groupby('Category')['Amount'].sum().reset_index()
                        if not expense_by_category.empty:
                            fig_expense = px.pie(
                                expense_by_category, 
                                values='Amount', 
                                names='Category',
                                hole=0.4,
                                color_discrete_sequence=px.colors.sequential.RdBu
                            )
                            fig_expense.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_expense, use_container_width=True)
                        else:
                            st.info("No expense data available by category.")
                    
                    with col2:
                        # Revenue Streams
                        st.markdown("#### 💵 Revenue Streams")
                        revenue_by_category = fin_df[fin_df['Type'] == 'Sale'].groupby('Category')['Amount'].sum().reset_index()
                        if not revenue_by_category.empty:
                            fig_revenue = px.bar(
                                revenue_by_category, 
                                x='Category', 
                                y='Amount',
                                color='Category',
                                labels={'Amount': 'Revenue (Kshs)', 'Category': 'Category'},
                                text_auto='.2s'
                            )
                            fig_revenue.update_traces(textposition='outside')
                            st.plotly_chart(fig_revenue, use_container_width=True)
                        else:
                            st.info("No revenue data available by category.")
                    
                    # Financial Health Indicators
                    st.markdown("### 📊 Financial Health")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Ratio", "2.5", "+0.2 from last month")
                    with col2:
                        st.metric("Quick Ratio", "1.8", "+0.1 from last month")
                    with col3:
                        st.metric("Debt-to-Equity", "0.4", "-0.1 from last month")
                
                with tab2:  # Trends Tab
                    st.markdown("### 📈 Financial Trends")
                    
                    # Time period selector
                    time_period = st.radio("View by:", ["Monthly", "Quarterly", "Yearly"], horizontal=True)
                    
                    # Prepare data based on time period
                    if time_period == "Monthly":
                        fin_df['Period'] = fin_df['Date'].dt.strftime('%Y-%m')
                    elif time_period == "Quarterly":
                        fin_df['Period'] = fin_df['Date'].dt.to_period('Q').astype(str)
                    else:  # Yearly
                        fin_df['Period'] = fin_df['Date'].dt.strftime('%Y')
                    
                    # Calculate trends
                    trends_data = fin_df.groupby(['Period', 'Type'])['Amount'].sum().unstack(fill_value=0)
                    for col in ['Sale', 'Expense']:
                        if col not in trends_data.columns:
                            trends_data[col] = 0
                    trends_data['Profit'] = trends_data['Sale'] - trends_data['Expense']
                    trends_data = trends_data.reset_index()
                    
                    # Create trend chart
                    fig_trend = px.line(
                        trends_data,
                        x='Period',
                        y=['Sale', 'Expense', 'Profit'],
                        title=f'{time_period} Financial Performance',
                        labels={'value': 'Amount (Kshs)', 'variable': 'Metric'},
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Profit Margin Trend
                    trends_data['Profit Margin'] = (trends_data['Profit'] / trends_data['Sale'] * 100).round(1)
                    fig_margin = px.line(
                        trends_data,
                        x='Period',
                        y='Profit Margin',
                        title=f'{time_period} Profit Margin Trend',
                        labels={'Profit Margin': 'Profit Margin (%)'},
                        template='plotly_white'
                    )
                    fig_margin.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_margin, use_container_width=True)
                
                with tab3:  # Details Tab
                    st.markdown("### 📋 Financial Details")
                    
                    # Date range filter
                    min_date = fin_df['Date'].min().date()
                    max_date = fin_df['Date'].max().date()
                    date_range = st.date_input(
                        "Select Date Range:",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        format="DD/MM/YYYY"
                    )
                    
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        filtered_df = fin_df[(fin_df['Date'].dt.date >= start_date) & 
                                          (fin_df['Date'].dt.date <= end_date)]
                        
                        # Transaction Summary
                        st.markdown("#### Transaction Summary")
                        summary_cols = st.columns(4)
                        with summary_cols[0]:
                            st.metric("Total Transactions", len(filtered_df))
                        with summary_cols[1]:
                            st.metric("Total Sales", f"Kshs {filtered_df[filtered_df['Type'] == 'Sale']['Amount'].sum():,.2f}")
                        with summary_cols[2]:
                            st.metric("Total Expenses", f"Kshs {filtered_df[filtered_df['Type'] == 'Expense']['Amount'].sum():,.2f}")
                        with summary_cols[3]:
                            profit = filtered_df[filtered_df['Type'] == 'Sale']['Amount'].sum() - filtered_df[filtered_df['Type'] == 'Expense']['Amount'].sum()
                            st.metric("Net Profit", f"Kshs {profit:,.2f}")
                        
                        # Detailed Transactions
                        st.markdown("#### Transaction Details")
                        display_cols = ['Date', 'Type', 'Category', 'Description', 'Hog ID', 'Amount']
                        if 'Hog ID' in filtered_df.columns:
                            filtered_df['Hog ID'] = filtered_df['Hog ID'].fillna('N/A')
                        st.dataframe(
                            filtered_df[display_cols].sort_values('Date', ascending=False),
                            column_config={
                                'Date': st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                                'Amount': st.column_config.NumberColumn("Amount (Kshs)", format="%.2f")
                            },
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Export button
                        csv = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Export to CSV",
                            data=csv,
                            file_name=f"financial_details_{start_date}_to_{end_date}.csv",
                            mime='text/csv'
                        )
            else:
                st.info("No financial data available. Add financial records to see insights.")
                
            # Add some spacing at the bottom
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Initialize monthly_data as None
            monthly_data = None
            
            # Check if we have financial data to process
            if not st.session_state['financial_data'].empty:
                fin_df = st.session_state['financial_data'].copy()
                fin_df['Date'] = pd.to_datetime(fin_df['Date'], errors='coerce')
                fin_df = fin_df.dropna(subset=['Date'])
                fin_df['Month'] = fin_df['Date'].dt.to_period('M')
                fin_df['Year'] = fin_df['Date'].dt.year
                
                # Calculate monthly data for detailed view
                if not fin_df.empty:
                    monthly_data = fin_df.groupby('Month').agg({
                        'Amount': lambda x: x[fin_df['Type'] == 'Sale'].sum(),
                        'Type': 'count'
                    }).rename(columns={'Amount': 'Sale', 'Type': 'Transaction Count'})
                    
                    # Add expenses
                    monthly_expenses = fin_df[fin_df['Type'] == 'Expense'].groupby('Month')['Amount'].sum()
                    monthly_data['Expense'] = monthly_expenses
                    monthly_data['Expense'] = monthly_data['Expense'].fillna(0)
                    
                    # Calculate profit and cumulative profit
                    monthly_data['Profit'] = monthly_data['Sale'] - monthly_data['Expense']
                    monthly_data['Cumulative Profit'] = monthly_data['Profit'].cumsum()
                    monthly_data = monthly_data.reset_index()
            
            # Add data table for detailed view
            with st.expander("📋 View Detailed Monthly Data"):
                if monthly_data is not None and not monthly_data.empty:
                    # Ensure all required columns exist
                    required_columns = ['Month', 'Sale', 'Expense', 'Profit', 'Cumulative Profit']
                    
                    # Create a new DataFrame with all required columns, filling missing ones with 0
                    display_data = pd.DataFrame(index=monthly_data.index)
                    for col in required_columns:
                        if col in monthly_data.columns:
                            display_data[col] = monthly_data[col]
                        else:
                            display_data[col] = 0
                
                # Rename columns for display
                display_data = display_data.rename(columns={
                    'Month': 'Month',
                    'Sale': 'Sales (Kshs)',
                    'Expense': 'Expenses (Kshs)',
                    'Profit': 'Monthly Profit (Kshs)',
                    'Cumulative Profit': 'Cumulative Profit (Kshs)'
                })
                
                # Format and display the DataFrame
                st.dataframe(
                            display_data.style.format({
                                'Sales (Kshs)': '{:,.0f}',
                                'Expenses (Kshs)': '{:,.0f}',
                                'Monthly Profit (Kshs)': '{:,.0f}',
                                'Cumulative Profit (Kshs)': '{:,.0f}'
                            }),
                            use_container_width=True,
                            height=min(400, 35 * len(display_data) + 30)  # Dynamic height based on number of rows
                        )
        
        with tab_finance:
            st.markdown("### 💰 Financial Summary")
            
            if not st.session_state['financial_data'].empty:
                fin_df = st.session_state['financial_data'].copy()
                fin_df['Date'] = pd.to_datetime(fin_df['Date'], errors='coerce')
                fin_df = fin_df.dropna(subset=['Date'])
                
                # Calculate key metrics
                total_sales = fin_df[fin_df['Type'] == 'Sale']['Amount'].sum()
                total_expenses = fin_df[fin_df['Type'] == 'Expense']['Amount'].sum()
                net_profit = total_sales - total_expenses
                gross_profit_margin = ((total_sales - total_expenses) / total_sales * 100) if total_sales > 0 else 0
                
                # Display key metrics with improved layout and full visibility
                st.markdown(f"""
                <style>
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background: #ffffff;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    text-align: center;
                    border-left: 5px solid #4CAF50;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    height: 100%;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    overflow: visible;
                }}
                .metric-card h3 {{
                    color: #555;
                    font-size: 1rem;
                    margin: 0 0 10px 0;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }}
                .metric-card h2 {{
                    color: #2c3e50;
                    font-size: 1.8rem;
                    margin: 5px 0;
                    font-weight: 700;
                    white-space: nowrap;
                    overflow: visible;
                    text-overflow: clip;
                }}
                .metric-card p {{
                    color: #7f8c8d;
                    font-size: 0.9rem;
                    margin: 5px 0 0 0;
                }}
                .metric-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
                }}
                /* Tooltip for full value on hover */
                .metric-tooltip {{
                    position: relative;
                    display: inline-block;
                    width: 100%;
                }}
                .metric-tooltip .tooltiptext {{
                    visibility: hidden;
                    width: auto;
                    background-color: #333;
                    color: #fff;
                    text-align: center;
                    border-radius: 6px;
                    padding: 5px 10px;
                    position: absolute;
                    z-index: 1;
                    bottom: 125%;
                    left: 50%;
                    transform: translateX(-50%);
                    opacity: 0;
                    transition: opacity 0.3s;
                    white-space: nowrap;
                    font-size: 0.9rem;
                }}
                .metric-tooltip:hover .tooltiptext {{
                    visibility: visible;
                    opacity: 1;
                }}
                @media (max-width: 768px) {{
                    .metrics-grid {{
                        grid-template-columns: 1fr 1fr;
                    }}
                }}
                @media (max-width: 480px) {{
                    .metrics-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
                
                    margin-bottom: 20px;
                    width: 100%;
                }}
                .metric-card {{
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 15px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    min-width: 0; /* Prevents overflow */
                }}
                .metric-title {{
                    font-size: 14px;
                    color: #6c757d;
                    margin-bottom: 8px;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }}
                .metric-value {{
                    font-size: 16px;
                    font-weight: 600;
                    color: #212529;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }}
                .metric-delta {{
                    font-size: 12px;
                    color: #28a745;
                    margin-top: 4px;
                }}
                .metric-delta.negative {{
                    color: #dc3545;
                }}
                @media (max-width: 992px) {{
                    .metrics-grid {{
                        grid-template-columns: repeat(2, 1fr);
                    }}
                }}
                @media (max-width: 576px) {{
                    .metrics-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
                </style>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Total Revenue</h3>
                        <div class="metric-tooltip">
                            <h2>Kshs {total_sales:,.2f}</h2>
                            <span class="tooltiptext">Kshs {total_sales:,.2f}</span>
                        </div>
                        <p>All-time sales revenue</p>
                    </div>
                    <div class="metric-card">
                        <h3>Total Expenses</h3>
                        <div class="metric-tooltip">
                            <h2>Kshs {total_expenses:,.2f}</h2>
                            <span class="tooltiptext">Kshs {total_expenses:,.2f}</span>
                        </div>
                        <p>All-time operational costs</p>
                    </div>
                    <div class="metric-card">
                        <h3>Net Profit</h3>
                        <div class="metric-tooltip">
                            <h2 style="color: {'#27ae60' if net_profit >= 0 else '#e74c3c'}">
                                {'+' if net_profit >= 0 else ''}Kshs {abs(net_profit):,.2f}
                            </h2>
                            <span class="tooltiptext">{'+' if net_profit >= 0 else ''}Kshs {abs(net_profit):,.2f}</span>
                        </div>
                        <p style="color: {'#27ae60' if net_profit >= 0 else '#e74c3c'}">
                            ({'+' if net_profit >= 0 else ''}{(net_profit/total_sales*100) if total_sales > 0 else 0:.1f}% margin)
                        </p>
                    </div>
                    <div class="metric-card">
                        <h3>Profit Margin</h3>
                        <div class="metric-tooltip">
                            <h2 style="color: {'#27ae60' if gross_profit_margin >= 0 else '#e74c3c'}">
                                {gross_profit_margin:,.1f}%
                            </h2>
                            <span class="tooltiptext">{gross_profit_margin:,.1f}%</span>
                        </div>
                        <p>{(total_sales - total_expenses):,.0f} / {total_sales:,.0f}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a divider and link to Insights tab
                st.markdown("---")
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h3>🔍 Want more detailed financial insights?</h3>
                    <p>Visit the <strong>Insights</strong> tab for comprehensive financial analysis including:</p>
                    <ul>
                        <li>Expense breakdown by category</li>
                        <li>Revenue streams analysis</li>
                        <li>Financial health indicators</li>
                        <li>Trend analysis over time</li>
                        <li>Detailed transaction history</li>
                    </ul>
                    <p>Click on the <strong>Insights</strong> tab at the top to explore more.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recent Transactions
                st.markdown("### Recent Transactions")
                recent_transactions = fin_df[['Date', 'Type', 'Category', 'Description', 'Amount']]\
                    .sort_values('Date', ascending=False).head(10)
                
                if not recent_transactions.empty:
                    st.dataframe(
                        recent_transactions,
                        column_config={
                            'Date': st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                            'Amount': st.column_config.NumberColumn("Amount (Kshs)", format="%.2f")
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No recent transactions found.")
                    
            else:
                st.info("No financial data available. Add financial records to see insights.")
                
                # Add call to action
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h3>📊 Ready to track your finances?</h3>
                    <p>Start by adding your financial records to unlock powerful insights into your hog farming business.</p>
                    <p>Once you add data, you'll be able to track revenues, expenses, and profitability.</p>
                </div>
                """, unsafe_allow_html=True)
        
    with tab_efficiency:
        st.markdown("### 🥗 Feed Conversion Ratio (FCR)")
        
        # Initialize display_data if not already defined
        if 'display_data' not in locals() or display_data is None or display_data.empty:
            display_data = pd.DataFrame()
            
        if not st.session_state['financial_data'].empty and not display_data.empty and 'Hog ID' in display_data.columns:
            feed_df = st.session_state['financial_data'].copy()
            feed_df = feed_df[feed_df['Category'] == 'Feed']
            total_feed = feed_df['Amount'].sum()
                    
            # Calculate total weight gain
            weight_gains = []
            valid_hogs = 0
            
            for hog_id in display_data['Hog ID'].unique():
                hog_data = display_data[display_data['Hog ID'] == hog_id].sort_values(by='Date')
                if len(hog_data) > 1 and 'Weight (kg)' in hog_data.columns:
                    try:
                        initial_weight = hog_data['Weight (kg)'].iloc[0]
                        final_weight = hog_data['Weight (kg)'].iloc[-1]
                        if pd.notna(initial_weight) and pd.notna(final_weight):
                            gain = final_weight - initial_weight
                            if gain > 0:  # Only include positive gains
                                weight_gains.append((hog_id, gain))
                                valid_hogs += 1
                    except (IndexError, KeyError):
                        continue  # Skip this hog if there's an error in weight data
            
            total_weight_gain = sum(gain for _, gain in weight_gains) if weight_gains else 0
            
            if total_weight_gain > 0 and total_feed > 0:
                fcr = total_feed / total_weight_gain
            else:
                fcr = None
                
            if valid_hogs == 0:
                st.warning("No valid hog weight data available for FCR calculation.")
                return
            
            # Display FCR metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Feed Consumed", f"{total_feed:,.1f} kg")
            with col2:
                st.metric("Total Weight Gain", f"{total_weight_gain:,.1f} kg")
            with col3:
                if fcr is not None:
                    st.metric("Feed Conversion Ratio (FCR)", f"{fcr:.2f}", 
                             help="Lower is better. Represents kg of feed per kg of weight gain")
                else:
                    st.metric("Feed Conversion Ratio (FCR)", "N/A", 
                             help="Insufficient data to calculate FCR")
            
            # Display FCR by hog
            st.markdown("### 🐖 FCR by Hog")
            hog_fcr = []
            for hog_id, gain in weight_gains:
                hog_feed = feed_df[feed_df['Hog ID'] == hog_id]['Amount'].sum()
                hog_fcr.append({
                    'Hog ID': hog_id,
                    'Feed Consumed (kg)': hog_feed,
                    'Weight Gain (kg)': gain,
                    'FCR': hog_feed / gain if gain > 0 else None
                })
                
                if hog_fcr:
                    try:
                        # Create DataFrame and handle potential data issues
                        fcr_df = pd.DataFrame(hog_fcr)
                        
                        # Ensure we have the required columns
                        required_cols = ['Hog ID', 'FCR']
                        if all(col in fcr_df.columns for col in required_cols):
                            # Sort by FCR, handling any None values
                            fcr_df = fcr_df.sort_values('FCR', na_position='last')
                            
                            # Create a bar chart for FCR by hog
                            fig = px.bar(
                                fcr_df,
                                x='Hog ID',
                                y='FCR',
                                title='Feed Conversion Ratio by Hog',
                                labels={'FCR': 'Feed Conversion Ratio (kg/kg)', 'Hog ID': 'Hog ID'},
                                color='FCR',
                                color_continuous_scale='RdYlGn_r',  # Red-Yellow-Green (reversed, so green is best)
                                text='FCR'
                            )
                            
                            fig.update_traces(
                                hovertemplate='<b>Hog %{x}</b><br>FCR: %{y:.2f} kg/kg<extra></extra>',
                                texttemplate='%{text:.1f}',
                                textposition='outside'
                            )
                            
                            fig.update_layout(
                                xaxis_title='Hog ID',
                                yaxis_title='Feed Conversion Ratio (kg/kg)',
                                coloraxis_showscale=False,
                                height=400,
                                yaxis=dict(range=[0, fcr_df['FCR'].max() * 1.1])  # Add some padding to the top
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add a small data table below the chart
                            st.markdown("#### Detailed FCR Data")
                            display_cols = ['Hog ID', 'Feed (kg)', 'Weight Gain (kg)', 'FCR']
                            display_df = fcr_df[display_cols].copy()
                            display_df['FCR'] = display_df['FCR'].round(2)
                            st.dataframe(
                                display_df.style.format({
                                    'Feed (kg)': '{:.1f}',
                                    'Weight Gain (kg)': '{:.1f}',
                                    'FCR': '{:.2f}'
                                }),
                                use_container_width=True,
                                height=min(300, 35 * len(display_df) + 30)  # Dynamic height
                            )
                        else:
                            st.warning("Required columns for FCR calculation are missing.")
                    except Exception as e:
                        st.error(f"Error creating FCR visualization: {str(e)}")
                else:
                    st.info("No valid hog FCR data available. Ensure you have:")
                    st.markdown("""
                    - Multiple weight recordings per hog
                    - Positive weight gains
                    - Valid feed data
                    """)
            else:
                st.info("No weight gain data available for FCR calculation.")
        else:
            st.info("No financial or weight data available for FCR analysis.")
        
        with tab_growth:
            st.markdown("### 📈 Week-to-Week Growth Analysis")
            
            if not display_data.empty and 'Hog ID' in display_data.columns and 'Weight (kg)' in display_data.columns and 'Date' in display_data.columns:
                # Calculate growth data
                growth_data = []
                valid_hogs = 0
                
                try:
                    # Get unique hog IDs, handling potential non-string values
                    hog_ids = [str(hid) for hid in display_data['Hog ID'].unique() if pd.notna(hid)]
                except Exception as e:
                    st.warning(f"Error processing hog IDs: {str(e)}")
                    hog_ids = []
                
                for hog_id in hog_ids:
                    try:
                        hog_data = display_data[display_data['Hog ID'].astype(str) == str(hog_id)].sort_values(by='Date')
                        if len(hog_data) > 1 and 'Weight (kg)' in hog_data.columns:
                            hog_data = hog_data.copy()
                            # Ensure we have valid weight data
                            if not hog_data['Weight (kg)'].isna().all():
                                hog_data['Previous Weight (kg)'] = hog_data['Weight (kg)'].shift(1)
                                # Only calculate if we have at least two valid weight measurements
                                if len(hog_data.dropna(subset=['Weight (kg)', 'Previous Weight (kg)'])) >= 2:
                                    hog_data['Weight Change (kg)'] = hog_data['Weight (kg)'] - hog_data['Previous Weight (kg)']
                                    hog_data['Growth (%)'] = (hog_data['Weight Change (kg)'] / hog_data['Previous Weight (kg)']) * 100
                                    
                                    # Only include hogs with valid growth data
                                    valid_growth = hog_data.dropna(subset=['Weight Change (kg)', 'Growth (%)'])
                                    if not valid_growth.empty:
                                        growth_data.append(valid_growth)
                                        valid_hogs += 1
                    except Exception as e:
                        st.warning(f"Error processing hog {hog_id}: {str(e)}")
                        continue
                
                if valid_hogs > 0 and growth_data:
                    try:
                        all_growth_data = pd.concat(growth_data)
                        
                        # Calculate summary metrics with error handling
                        avg_growth_percent = all_growth_data['Growth (%)'].mean() if 'Growth (%)' in all_growth_data.columns else 0
                        num_hogs = all_growth_data['Hog ID'].nunique() if 'Hog ID' in all_growth_data.columns else 0
                        
                        # Safely calculate positive/negative growth counts
                        if 'Weight Change (kg)' in all_growth_data.columns and 'Hog ID' in all_growth_data.columns:
                            num_positive = all_growth_data[all_growth_data['Weight Change (kg)'] > 0]['Hog ID'].nunique()
                            num_negative = all_growth_data[all_growth_data['Weight Change (kg)'] < 0]['Hog ID'].nunique()
                        else:
                            num_positive = num_negative = 0
                    except Exception as e:
                        st.error(f"Error calculating growth metrics: {str(e)}")
                        return
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Average Weekly Growth", f"{avg_growth_percent:.1f}%")
                    with col2:
                        st.metric("Hogs Tracked", num_hogs)
                    with col3:
                        st.metric("Hogs Gaining Weight", f"{num_positive} ({num_positive/num_hogs*100:.0f}%)")
                    with col4:
                        st.metric("Hogs Losing Weight", f"{num_negative} ({num_negative/num_hogs*100:.0f}%)")
                    
                    # Interactive Growth Trends Section
                    st.markdown("#### 📊 Interactive Growth Trends")
                    
                    # Create a copy of the data for manipulation
                    growth_df = all_growth_data.copy()
                    
                    # Get unique hogs and sort them
                    unique_hogs = sorted(growth_df['Hog ID'].unique())
                    
                    # Create a multi-select widget for hogs
                    selected_hogs = st.multiselect(
                        'Select Hogs to Compare:',
                        options=unique_hogs,
                        default=unique_hogs[:min(3, len(unique_hogs))],  # Show first 3 hogs by default
                        format_func=lambda x: f"Hog {x}",
                        key='hog_selector_growth'
                    )
                    
                    # Filter data for selected hogs
                    if selected_hogs:
                        filtered_data = growth_df[growth_df['Hog ID'].isin(selected_hogs)]
                        
                        # Create the main line plot
                        fig = px.line(
                            filtered_data,
                            x='Date',
                            y='Growth (%)',
                            color='Hog ID',
                            title=f'Weekly Growth Rate for Selected Hogs',
                            labels={'Growth (%)': 'Growth (%)', 'Date': 'Date'},
                            template='plotly_white',
                            color_discrete_sequence=px.colors.qualitative.Plotly,
                            hover_data={'Hog ID': True, 'Weight (kg)': ':.1f', 'Weight Change (kg)': ':.1f'}
                        )
                        
                        # Add a horizontal line at 0% growth
                        fig.add_hline(y=0, line_dash="dash", line_color="red", 
                                    annotation_text="No Growth", 
                                    annotation_position="bottom right")
                        
                        # Add target growth range (2-5%)
                        fig.add_hrect(y0=2, y1=5, line_width=0, fillcolor="green", opacity=0.1,
                                    annotation_text="Target Growth Range", annotation_position="top left")
                        
                        # Update layout for better readability
                        fig.update_layout(
                            xaxis_title='Date',
                            yaxis_title='Weekly Growth (%)',
                            height=500,
                            showlegend=True,
                            legend_title_text='Hog ID',
                            hovermode='x unified',
                            margin=dict(t=50, b=50, l=50, r=50),
                            xaxis=dict(rangeslider_visible=True)  # Add range slider for better navigation
                        )
                        
                        # Customize hover template
                        fig.update_traces(
                            hovertemplate='''
                            <b>Hog %{customdata[0]}</b><br>
                            Date: %{x|%Y-%m-%d}<br>
                            Growth: %{y:.1f}%<br>
                            Weight: %{customdata[1]:.1f} kg<br>
                            Change: %{customdata[2]:+.1f} kg
                            <extra></extra>
                            ''',
                            line=dict(width=2.5)
                        )
                        
                        # Display the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add a small table showing the latest growth data for selected hogs
                        st.markdown("#### 📋 Latest Growth Data")
                        latest_data = filtered_data.sort_values('Date').groupby('Hog ID').last().reset_index()
                        
                        # Format the table
                        st.dataframe(
                            latest_data[['Hog ID', 'Date', 'Weight (kg)', 'Weight Change (kg)', 'Growth (%)']]
                                .sort_values('Growth (%)', ascending=False)
                                .style
                                .format({
                                    'Weight (kg)': '{:.1f} kg',
                                    'Weight Change (kg)': '{:+.1f} kg',
                                    'Growth (%)': '{:+.1f}%'
                                })
                                .apply(lambda x: ['background-color: #e6f7e6' if x['Growth (%)'] > 0 else 'background-color: #ffebee' 
                                                for _ in x], axis=1)
                                .set_properties(**{'text-align': 'center'}),
                            use_container_width=True,
                            height=min(300, 45 * len(selected_hogs) + 45)
                        )
                        
                    else:
                        st.info("Please select at least one hog to view growth trends.")
                    
                    # Growth summary by hog
                    st.markdown("#### Growth Summary by Hog")
                    
                    # Calculate growth metrics per hog
                    hog_summary = []
                    for hog_id in all_growth_data['Hog ID'].unique():
                        hog_data = all_growth_data[all_growth_data['Hog ID'] == hog_id]
                        if len(hog_data) > 0:
                            hog_summary.append({
                                'Hog ID': hog_id,
                                'Avg Weekly Growth (%)': hog_data['Growth (%)'].mean(),
                                'Total Weight Gain (kg)': hog_data['Weight Change (kg)'].sum(),
                                'Weeks Tracked': len(hog_data)
                            })
                    
                    if hog_summary:
                        summary_df = pd.DataFrame(hog_summary).sort_values('Avg Weekly Growth (%)', ascending=False)
                        
                        # Display metrics in a table with conditional formatting
                        st.dataframe(
                            summary_df.style
                                .format({
                                    'Avg Weekly Growth (%)': '{:.1f}%',
                                    'Total Weight Gain (kg)': '{:.1f} kg',
                                    'Weeks Tracked': '{:d}'
                                })
                                .background_gradient(
                                    subset=['Avg Weekly Growth (%)'], 
                                    cmap='RdYlGn',
                                    vmin=summary_df['Avg Weekly Growth (%)'].min(),
                                    vmax=summary_df['Avg Weekly Growth (%)'].max()
                                )
                                .set_properties(**{'text-align': 'center'}),
                            use_container_width=True,
                            height=min(400, len(summary_df) * 35 + 50)  # Dynamic height based on rows
                        )
                        
                        # Top Performers Section
                        st.markdown("### 🏆 Top Performers")
                        
                        # Get top 3 performers by growth rate
                        top_performers = summary_df.nlargest(3, 'Avg Weekly Growth (%)')
                        
                        if not top_performers.empty:
                            cols = st.columns(min(3, len(top_performers)))
                            for idx, (_, row) in enumerate(top_performers.iterrows()):
                                with cols[idx % 3]:
                                    try:
                                        hog_id = int(float(row['Hog ID']))  # Convert to float first to handle string numbers
                                        st.markdown(
                                            f"<div class='metric-card' style='background-color: #e8f5e9;'>"
                                            f"<h3>🥇 Hog #{hog_id:03d}</h3>"
                                            f"<h2>{row['Avg Weekly Growth (%)']:.1f}%</h2>"
                                            f"<p>+{row['Total Weight Gain (kg)']:.1f} kg over {row['Weeks Tracked']} weeks</p>"
                                            "</div>",
                                            unsafe_allow_html=True
                                        )
                                    except (ValueError, TypeError):
                                        st.markdown(
                                            f"<div class='metric-card' style='background-color: #e8f5e9;'>"
                                            f"<h3>🥇 Hog {row['Hog ID']}</h3>"
                                            f"<h2>{row['Avg Weekly Growth (%)']:.1f}%</h2>"
                                            f"<p>+{row['Total Weight Gain (kg)']:.1f} kg over {row['Weeks Tracked']} weeks</p>"
                                            "</div>",
                                            unsafe_allow_html=True
                                        )
                        
                        # Potential Issues Section
                        st.markdown("### ⚠️ Potential Issues")
                        
                        # Hogs with no weight gain for two consecutive weeks
                        all_growth_data = pd.concat(growth_data)
                        all_growth_data['No Gain Flag'] = (all_growth_data['Weight Change (kg)'] <= 0).rolling(window=2).sum() == 2
                        no_gain_hogs = all_growth_data[all_growth_data['No Gain Flag'] == True]
                        
                        if not no_gain_hogs.empty:
                            st.warning("#### 🚨 Hogs with No Weight Gain (Two Consecutive Weeks)")
                            no_gain_summary = no_growth_summary(no_gain_hogs)
                            st.dataframe(
                                no_gain_summary.style
                                    .format({
                                        'Weight (kg)': '{:.1f} kg',
                                        'Growth (%)': '{:.1f}%',
                                        'Weeks Without Gain': '{:d}'
                                    })
                                    .background_gradient(
                                        subset=['Weeks Without Gain'], 
                                        cmap='YlOrRd',
                                        vmin=1,
                                        vmax=no_gain_summary['Weeks Without Gain'].max()
                                    )
                                    .set_properties(**{'text-align': 'center'}),
                                use_container_width=True
                            )
                        else:
                            st.success("✅ No hogs detected with no weight gain for two consecutive weeks.")
                        
                        # Hogs with negative growth
                        negative_growth = all_growth_data[all_growth_data['Weight Change (kg)'] < 0]
                        if not negative_growth.empty:
                            st.warning("#### 📉 Hogs with Negative Growth")
                            
                            # Get the most recent records for each hog with negative growth
                            recent_neg_growth = negative_growth.sort_values(['Hog ID', 'Date']).drop_duplicates('Hog ID', keep='last')
                            
                            # Display metrics
                            neg_col1, neg_col2 = st.columns(2)
                            with neg_col1:
                                st.metric("Hogs with Negative Growth", len(recent_neg_growth))
                            with neg_col2:
                                avg_neg_growth = recent_neg_growth['Growth (%)'].mean()
                                st.metric("Average Negative Growth", f"{avg_neg_growth:.1f}%")
                            
                            # Show details in an expander
                            with st.expander("🔍 View Details"):
                                st.dataframe(
                                    recent_neg_growth[['Hog ID', 'Date', 'Weight (kg)', 'Weight Change (kg)', 'Growth (%)']]
                                        .rename(columns={
                                            'Hog ID': 'Hog ID',
                                            'Date': 'Last Record',
                                            'Weight (kg)': 'Current Weight (kg)',
                                            'Weight Change (kg)': 'Loss (kg)',
                                            'Growth (%)': 'Growth (%)'
                                        })
                                        .sort_values('Growth (%)')
                                        .style
                                            .format({
                                                'Current Weight (kg)': '{:.1f} kg',
                                                'Loss (kg)': '{:.1f} kg',
                                                'Growth (%)': '{:.1f}%'
                                            })
                                            .background_gradient(
                                                subset=['Growth (%)'], 
                                                cmap='Reds_r',
                                                vmin=recent_neg_growth['Growth (%)'].min(),
                                                vmax=0
                                            )
                                            .set_properties(**{'text-align': 'center'}),
                                    use_container_width=True,
                                    height=min(300, len(recent_neg_growth) * 35 + 50)
                                )
                        else:
                            st.success("✅ No hogs with negative growth detected.")
                        
                        # Enhanced Outlier Detection Section
                        st.markdown("### 📊 Advanced Outlier Analysis")
                        
                        if not all_growth_data.empty and len(all_growth_data) > 5:  # Need sufficient data
                            # Add user controls for outlier detection
                            st.markdown("#### ⚙️ Detection Settings")
                            col1, col2 = st.columns(2)
                            with col1:
                                z_threshold = st.slider(
                                    "Sensitivity (standard deviations)", 
                                    min_value=1.5, 
                                    max_value=3.0, 
                                    value=2.0, 
                                    step=0.1,
                                    help="Lower values detect more potential outliers"
                                )
                            with col2:
                                min_growth = st.number_input(
                                    "Minimum growth rate to flag (%)",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=20.0,
                                    step=0.5,
                                    help="Flag growth rates above this value as potential data errors"
                                )
                            
                            # Calculate z-scores for growth rates
                            growth_rates = all_growth_data['Growth (%)'].dropna()
                            
                            if len(growth_rates) > 1:  # Need at least 2 data points
                                from scipy import stats
                                z_scores = stats.zscore(growth_rates)
                                abs_z_scores = np.abs(z_scores)
                                
                                # Identify statistical outliers
                                stat_outliers = all_growth_data.iloc[np.where(abs_z_scores > z_threshold)[0]]
                                
                                # Identify unusually high growth rates (potential data errors)
                                high_growth = all_growth_data[all_growth_data['Growth (%)'] > min_growth]
                                
                                if not stat_outliers.empty or not high_growth.empty:
                                    # Create a single DataFrame for all potential issues
                                    all_issues = pd.concat([
                                        stat_outliers.assign(Issue_Type='Statistical Outlier'),
                                        high_growth.assign(Issue_Type='High Growth Rate')
                                    ]).drop_duplicates()
                                    
                                    # Add visualization
                                    st.markdown("#### 📈 Growth Rate Overview")
                                    
                                    # Create a simpler histogram with a KDE line
                                    fig = px.histogram(
                                        all_growth_data, 
                                        x='Growth (%)',
                                        title='Distribution of Weekly Growth Rates',
                                        nbins=20,
                                        color_discrete_sequence=['#4CAF50'],
                                        opacity=0.8,
                                        marginal='box',
                                        template='plotly_white'
                                    )
                                    
                                    # Add vertical lines for key metrics
                                    mean_growth = growth_rates.mean()
                                    median_growth = growth_rates.median()
                                    
                                    # Add mean and median lines
                                    fig.add_vline(
                                        x=mean_growth,
                                        line_dash="dash",
                                        line_color="blue",
                                        annotation_text=f"Mean: {mean_growth:.1f}%",
                                        annotation_position="top right"
                                    )
                                    
                                    fig.add_vline(
                                        x=median_growth,
                                        line_dash="solid",
                                        line_color="purple",
                                        annotation_text=f"Median: {median_growth:.1f}%",
                                        annotation_position="top left"
                                    )
                                    
                                    # Add target growth range
                                    target_min, target_max = 2.0, 5.0  # Example target growth range
                                    fig.add_vrect(
                                        x0=target_min, x1=target_max,
                                        fillcolor="green", opacity=0.1,
                                        annotation_text=f"Target Range: {target_min}-{target_max}%",
                                        annotation_position="top left"
                                    )
                                    
                                    # Update layout for better readability
                                    fig.update_layout(
                                        height=400,
                                        xaxis_title='Weekly Growth (%)',
                                        yaxis_title='Number of Measurements',
                                        showlegend=False,
                                        margin=dict(t=50, b=50, l=50, r=50),
                                        hovermode='x unified'
                                    )
                                    
                                    # Add summary statistics
                                    stats_text = f"""
                                    **Summary Statistics**  
                                    • Mean Growth: {mean_growth:.1f}%  
                                    • Median Growth: {median_growth:.1f}%  
                                    • Target Range: {target_min}-{target_max}%  
                                    • Total Measurements: {len(all_growth_data)}
                                    """
                                    st.markdown(stats_text)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show detailed table of potential issues
                                    st.markdown("#### ⚠️ Potential Issues Detected")
                                    
                                    # Add context to the issues
                                    all_issues = all_issues.sort_values(['Hog ID', 'Date'])
                                    
                                    # Add context columns
                                    all_issues['Context'] = all_issues.apply(
                                        lambda x: f"{x['Weight (kg)']:.1f}kg (Δ{x['Weight Change (kg)']:+.1f}kg, {x['Growth (%)']:.1f}%)",
                                        axis=1
                                    )
                                    
                                    # Create a clean DataFrame with the columns we need
                                    display_df = all_issues[[
                                        'Hog ID', 'Date', 'Weight (kg)', 
                                        'Weight Change (kg)', 'Growth (%)', 'Issue_Type'
                                    ]].copy()
                                    
                                    # Reset index to ensure it's clean and unique
                                    display_df = display_df.reset_index(drop=True)
                                    
                                    # Define thresholds
                                    ABNORMAL_GROWTH_THRESHOLD = 40  # 40% growth per week is considered abnormal
                                    
                                    # Filter to show only hogs with declines or abnormal growth
                                    display_df = display_df[
                                        (display_df['Growth (%)'] < 0) |  # All declines
                                        (display_df['Growth (%)'] > ABNORMAL_GROWTH_THRESHOLD)  # Abnormal growth
                                    ]
                                    
                                    # Create a sort key: 
                                    # - Declines first (sorted by growth %, most negative first)
                                    # - Then abnormal growth (sorted by growth %, highest first)
                                    display_df['_sort_key'] = display_df['Growth (%)'].apply(
                                        lambda x: (0, x) if x < 0 else (1, -x)
                                    )
                                    
                                    # Sort by the temporary key
                                    display_df = display_df.sort_values('_sort_key')
                                    display_df = display_df.drop('_sort_key', axis=1)
                                    
                                    # Add trend indicators and color coding
                                    def get_trend_icon(growth):
                                        if growth > ABNORMAL_GROWTH_THRESHOLD:  # Abnormal growth
                                            return '⚠️', '#fff3e6', '#cc8400', f'Abnormal Growth ({growth:+.1f}%)'
                                        elif growth < 0:  # Any decline
                                            return '📉', '#ffebee', '#c62828', f'Decline ({growth:+.1f}%)'
                                        return '➡️', '#e2e3e5', '#383d41', 'No Significant Change'
                                    
                                    # Apply formatting and styling
                                    trend_data = display_df['Growth (%)'].apply(get_trend_icon)
                                    display_df['Trend'] = trend_data.apply(lambda x: x[0])
                                    display_df['Status'] = trend_data.apply(lambda x: x[3])  # Add status text
                                    
                                    # Reorder columns to put Trend and Status first
                                    cols = ['Trend', 'Status'] + [col for col in display_df.columns if col not in ['Trend', 'Status']]
                                    display_df = display_df[cols]
                                    
                                    # Store the numeric values before formatting
                                    numeric_growth = display_df['Growth (%)'].copy()
                                    
                                    # Format the numeric columns for display
                                    display_df['Weight (kg)'] = display_df['Weight (kg)'].apply(lambda x: f"{float(x):.1f} kg")
                                    display_df['Weight Change (kg)'] = display_df['Weight Change (kg)'].apply(
                                        lambda x: f"{float(x):+.1f} kg"
                                    )
                                    display_df['Growth (%)'] = display_df['Growth (%)'].apply(
                                        lambda x: f"{float(x):+.1f}%" if float(x) != 0 else "0.0%"
                                    )
                                    
                                    # Create a styled DataFrame
                                    try:
                                        # Create a mapping of original indices to growth values
                                        growth_dict = dict(zip(display_df.index, numeric_growth))
                                        
                                        def style_row(row):
                                            # Get the growth value for this row using the index
                                            growth = growth_dict[row.name]
                                            _, bg_color, text_color, _ = get_trend_icon(growth)
                                            
                                            # Apply styles to all columns
                                            return [f'background-color: {bg_color}; color: {text_color}'] * len(row)
                                        
                                        # Apply row-based styling
                                        styled_df = display_df.style.apply(style_row, axis=1, subset=display_df.columns[2:])  # Skip Trend and Status columns
                                        
                                        # Display with improved formatting
                                        st.dataframe(
                                            styled_df,
                                            use_container_width=True,
                                            height=min(400, len(display_df) * 40 + 50),
                                            column_config={
                                                'Trend': st.column_config.TextColumn(
                                                    ' ',  # Empty header for the icon column
                                                    width='small',
                                                    help='Status indicators: ⚠️ = Abnormal Growth (>40%), 📉 = Any Decline, ➡️ = No Change'
                                                ),
                                                'Status': st.column_config.TextColumn(
                                                    'Status',
                                                    width='medium',
                                                    help='Status of the hog\'s growth pattern'
                                                ),
                                                'Hog ID': st.column_config.NumberColumn(
                                                    'Hog ID',
                                                    width='small'
                                                ),
                                                'Date': st.column_config.DateColumn(
                                                    'Date',
                                                    format='YYYY-MM-DD',
                                                    width='small'
                                                ),
                                                'Weight (kg)': st.column_config.NumberColumn(
                                                    'Weight (kg)',
                                                    format='%.1f',
                                                    width='small'
                                                ),
                                                'Weight Change (kg)': st.column_config.NumberColumn(
                                                    'Change (kg)',
                                                    format='%+.1f',
                                                    width='small'
                                                ),
                                                'Growth (%)': st.column_config.NumberColumn(
                                                    'Growth %',
                                                    format='%+.1f',
                                                    width='small'
                                                ),
                                                'Issue_Type': st.column_config.TextColumn(
                                                    'Issue Type',
                                                    width='medium'
                                                )
                                            },
                                            hide_index=True
                                        )
                                        
                                    except Exception as e:
                                        st.error(f"Error applying styles: {str(e)}")
                                        # Fallback to simple display without any styling
                                        st.dataframe(
                                            display_df,
                                            use_container_width=True,
                                            height=min(400, len(display_df) * 40 + 50)
                                        )
                                    
                                    # Add explanation and recommendations
                                    with st.expander("ℹ️ Understanding These Results"):
                                        st.markdown("""
                                        **About the detected issues:**
                                        
                                        - **Statistical Outliers**: Growth rates that are unusually high or low compared to the group
                                        - **High Growth Rates**: Growth rates exceeding the specified threshold (potential data errors)
                                        
                                        **Recommended Actions:**
                                        1. Review the highlighted records for data entry errors
                                        2. Check for measurement inconsistencies
                                        3. Investigate any health or environmental factors
                                        4. Adjust the sensitivity settings if needed
                                        """)
                                else:
                                    st.success("✅ No unusual growth patterns detected with current settings.")
                            else:
                                st.info("⚠️ Not enough data points for statistical analysis.")
                        elif not all_growth_data.empty:
                            st.info("ℹ️ More data points are needed for reliable outlier detection.")
                    
                    # Initialize all_growth_data as empty DataFrame if not already defined
                    if 'all_growth_data' not in locals():
                        all_growth_data = pd.DataFrame()
                    
                    # Skip the rest of the block if no data
                    display_growth_data = None
                else:
                    st.info("No weight data available for growth analysis.")
                    all_growth_data = pd.DataFrame()
                    display_growth_data = None
                                # Initialize variables with default values
                display_growth_data = None
                selection_options = ["Select a Hog ID"]  # Default empty selection
                
                # Only proceed if we have valid growth data
                if not all_growth_data.empty:
                    try:
                        display_growth_data = all_growth_data.copy()
                        # Ensure Hog ID is in the correct format
                        if 'Hog ID' in display_growth_data.columns:
                            display_growth_data['Hog ID'] = display_growth_data['Hog ID'].astype(str).apply(
                                lambda x: f'{int(float(x)):03d}' if str(x).isdigit() else x
                            )
                            unique_hog_ids_growth = sorted(display_growth_data['Hog ID'].unique())
                            # Add hog IDs to selection options if we have any
                            if unique_hog_ids_growth:
                                selection_options = ["Select a Hog ID"] + list(unique_hog_ids_growth)
                            else:
                                st.warning("No valid Hog IDs found in the growth data.")
                        else:
                            st.warning("No 'Hog ID' column found in the growth data.")
                    except Exception as e:
                        st.error(f"Error processing growth data: {str(e)}")
                else:
                    st.info("No growth data available for detailed view.")
                
                # Store in session state for persistence
                st.session_state['display_growth_data'] = display_growth_data

        # Ensure selection_options is defined
        if 'selection_options' not in locals():
            selection_options = ["Select a Hog ID"]
            
        # Only show the selectbox if we have options beyond the default
        if len(selection_options) > 1:
            selected_hog_id_for_growth = st.selectbox(
                "Select Hog ID to view detailed growth data:",
                options=selection_options,
                index=0,  # Default to the placeholder
                key='detailed_growth_hog_select'
            )
        else:
            selected_hog_id_for_growth = "Select a Hog ID"
            st.info("No hog data available for selection.")

            if selected_hog_id_for_growth != "Select a Hog ID" and display_growth_data is not None and 'Hog ID' in display_growth_data.columns:
                try:
                    filtered_detailed_growth = display_growth_data[
                        display_growth_data['Hog ID'].astype(str) == str(selected_hog_id_for_growth)
                    ]
                    
                    if not filtered_detailed_growth.empty:
                        # Select and format columns for display
                        display_columns = ['Hog ID', 'Date', 'Weight (kg)']
                        if 'Weight Change (kg)' in filtered_detailed_growth.columns:
                            display_columns.append('Weight Change (kg)')
                        if 'Growth (%)' in filtered_detailed_growth.columns:
                            display_columns.append('Growth (%)')
                        
                        # Format the DataFrame for display
                        display_df = filtered_detailed_growth[display_columns].copy()
                        
                        # Apply formatting
                        st.dataframe(
                            display_df.style.format({
                                'Weight (kg)': '{:.1f}',
                                'Weight Change (kg)': '{:.1f}',
                                'Growth (%)': '{:.1f}%'
                            } if 'Weight Change (kg)' in display_columns or 'Growth (%)' in display_columns else None),
                            hide_index=True, 
                            use_container_width=True
                        )
                    else:
                        st.info(f"No detailed growth data found for Hog {selected_hog_id_for_growth}.")
                except Exception as e:
                    st.error(f"Error displaying growth data: {str(e)}")

            # Plot Weight Trend for the selected Hog ID
            if selected_hog_id_for_growth != "Select a Hog ID":
                hog_data_for_plot = display_growth_data[display_growth_data['Hog ID'] == selected_hog_id_for_growth].sort_values(by='Date')
                if not hog_data_for_plot.empty and hog_data_for_plot['Weight (kg)'].nunique() > 1:
                    st.subheader(f"Weight Trend for Hog {selected_hog_id_for_growth}")
                    
                    # Use Plotly for interactive plot with hover
                    fig_plotly = px.line(hog_data_for_plot,
                                        x='Date',
                                        y='Weight (kg)',
                                        title=f"Weight Trend for Hog {selected_hog_id_for_growth}",
                                        markers=True,
                                        hover_data={'Date':'%d/%m/%Y', 'Weight (kg)':':.1fkg'})
                    
                    fig_plotly.update_traces(hovertemplate='Date: %{x}<br>Weight: %{y:.1f}kg')
                    fig_plotly.update_xaxes(tickangle=45)

                    st.plotly_chart(fig_plotly, use_container_width=True)
                    
                elif hog_data_for_plot['Weight (kg)'].nunique() <= 1:
                    st.info(f"Not enough data to plot trend for Hog {selected_hog_id_for_growth}. Need at least two different weight entries.")
                else:
                    st.info(f"No weight data available for Hog {selected_hog_id_for_growth} to plot trend.")

        st.markdown("--- ") # Separator
        # Outlier detection (simple example: using standard deviation)
        st.subheader("Outlier Detection")
        with st.expander("View Outlier Detection"):
            # Initialize all_growth_data if not already defined
            if 'all_growth_data' not in locals() or all_growth_data is None:
                all_growth_data = pd.DataFrame()
                
            if not all_growth_data.empty and 'Weight Change (kg)' in all_growth_data.columns and all_growth_data['Weight Change (kg)'].std() > 0:
                mean_growth = all_growth_data['Weight Change (kg)'].mean()
                std_growth = all_growth_data['Weight Change (kg)'].std()
                
                # Define an outlier as more than 2 standard deviations from the mean
                all_growth_data['Is Outlier'] = (
                    (all_growth_data['Weight Change (kg)'] > mean_growth + 2 * std_growth) |
                    (all_growth_data['Weight Change (kg)'] < mean_growth - 2 * std_growth)
                )
                
                outliers = all_growth_data[all_growth_data['Is Outlier']].copy() # Make a copy
                if not outliers.empty:
                    outliers['Hog ID'] = outliers['Hog ID'].apply(format_hog_id)
                    st.warning("Hogs with significant weight change (Outliers):")
                    st.dataframe(outliers[['Hog ID', 'Date', 'Weight (kg)', 'Weight Change (kg)', 'Growth (%)']], hide_index=True, use_container_width=True)
                else:
                    st.info("No significant outliers detected in weight change this week.")
            else:
                st.info("Not enough data or variation to detect outliers effectively.")

        st.markdown("--- ") # Separator
        # The filter data section should only be once in the sidebar
        # These filtering operations should apply to the displayed data in the main area of tab2
        filtered_data = display_data.copy() if 'display_data' in locals() and display_data is not None else pd.DataFrame()
        if search_hog_id:
            # Filter by formatted Hog ID if user inputs formatted string
            filtered_data['Formatted Hog ID'] = filtered_data['Hog ID'].apply(format_hog_id)
            filtered_data = filtered_data[filtered_data['Formatted Hog ID'].str.contains(search_hog_id, case=False)]
            filtered_data.drop(columns=['Formatted Hog ID'], inplace=True)

        if search_start_date:
            filtered_data = filtered_data[filtered_data['Date'] >= search_start_date]
        if search_end_date:
            filtered_data = filtered_data[filtered_data['Date'] <= search_end_date]
        
        if search_hog_id or search_start_date or search_end_date:
            st.subheader("Filtered Data") # Moved header inside conditional
            if not filtered_data.empty:
                display_filtered_data = filtered_data.copy()
                display_filtered_data['Hog ID'] = display_filtered_data['Hog ID'].apply(lambda x: f'{int(x):03d}')
                st.dataframe(display_filtered_data, hide_index=True, use_container_width=True) # Hide index
            else:
                st.info("No data matching your filter criteria.")

    with tab3: # Corrected indentation for tab3
        # Intelligent Summary Report
        st.subheader("Intelligent Summary Report")
        # Ensure display_data and all_growth_data are available for generate_summary_report
        # These are computed within tab1 and tab2, so we need to ensure they are available here.
        # A simpler way for now is to compute them once if they are not session_state variables.
        # Or, we can ensure the data is passed correctly.
        # For now, let's assume display_data and all_growth_data are defined in the broader scope or re-calculated.
        current_hog_data = st.session_state['hog_data']
        display_data_summary = current_hog_data.dropna(subset=['Hog ID', 'Date', 'Weight (kg)'])

        # Calculate growth data summary
        growth_data_summary = []
        if not display_data_summary.empty:
            # Ensure Date column is datetime
            display_data_summary['Date'] = pd.to_datetime(display_data_summary['Date'])
            
            for hog_id in display_data_summary['Hog ID'].unique():
                hog_data = display_data_summary[display_data_summary['Hog ID'] == hog_id].sort_values('Date')
                if len(hog_data) > 1:
                    hog_data = hog_data.copy()
                    hog_data['Previous Weight'] = hog_data['Weight (kg)'].shift(1)
                    hog_data['Weight Change (kg)'] = hog_data['Weight (kg)'] - hog_data['Previous Weight']
                    
                    # Calculate days between measurements
                    date_diffs = (hog_data['Date'] - hog_data['Date'].shift(1))
                    hog_data['Days Between'] = date_diffs.dt.days.fillna(0).astype(int)
                    
                    # Calculate daily gain, avoiding division by zero
                    hog_data['Daily Gain (kg/day)'] = hog_data.apply(
                        lambda x: x['Weight Change (kg)'] / x['Days Between'] if x['Days Between'] > 0 else 0,
                        axis=1
                    )
                    
                    # Calculate growth percentage
                    hog_data['Growth (%)'] = hog_data.apply(
                        lambda x: (x['Weight Change (kg)'] / x['Previous Weight']) * 100 if x['Previous Weight'] > 0 else 0,
                        axis=1
                    )
                    
                    # Add week information
                    hog_data['Week'] = hog_data['Date'].dt.strftime('%Y-W%U')
                    
                    growth_data_summary.append(hog_data.dropna(subset=['Weight Change (kg)', 'Daily Gain (kg/day)', 'Growth (%)', 'Week']))
        
        all_growth_data_summary = pd.concat(growth_data_summary) if growth_data_summary else pd.DataFrame()

        if not display_data_summary.empty:
            # Generate the summary report
            summary_report = generate_summary_report(display_data_summary, all_growth_data_summary)
            
            # Create columns for metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Hogs", 
                         summary_report['overall']['total_hogs'],
                         help="Total number of hogs being tracked")
            
            if 'average_daily_gain' in summary_report['overall']:
                with col2:
                    st.metric("Avg Daily Gain", 
                             f"{summary_report['overall']['average_daily_gain']:.2f} kg",
                             help="Average daily weight gain across all hogs")
            
            # Calculate days of data
            if not display_data.empty and 'Date' in display_data.columns and not display_data['Date'].isnull().all():
                try:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(display_data['Date']):
                        display_data['Date'] = pd.to_datetime(display_data['Date'], errors='coerce')
                    
                    # Filter out any NaT values that might have been created during conversion
                    valid_dates = display_data['Date'].dropna()
                    
                    if not valid_dates.empty:
                        date_range = (valid_dates.max() - valid_dates.min()).days
                        with col3:
                            st.metric("Tracking Period", 
                                     f"{date_range} days",
                                     help="Duration of weight tracking data")
                    else:
                        with col3:
                            st.metric("Tracking Period", 
                                     "N/A",
                                     help="No valid date data available")
                except Exception as e:
                    with col3:
                        st.metric("Tracking Period", 
                                 "Error",
                                 help="Error calculating date range")
                    st.error(f"Error processing date data: {str(e)}")
            
            # Count measurements
            with col4:
                st.metric("Total Measurements", 
                         len(display_data),
                         help="Total number of weight measurements recorded")
        
            # --- Performance Overview ---
            st.markdown("---")
            st.header("🏆 Performance Leaders")
            
            if 'best_overall_performer' in summary_report['overall'] and 'least_overall_performer' in summary_report['overall']:
                best = summary_report['overall']['best_overall_performer']
                least = summary_report['overall']['least_overall_performer']
                
                perf_col1, perf_col2 = st.columns(2)
                
                with perf_col1:
                    st.markdown("### 🥇 Top Performer")
                    st.metric("Hog ID", 
                             f"#{format_hog_id(best['id'])}", 
                             f"{best['adg']:.2f} kg/day")
                    st.progress(min(1.0, best['adg'] / 1.0), 
                               text=f"{best['adg']:.2f} kg/day")
                
                with perf_col2:
                    st.markdown("### ⚠️ Needs Attention")
                    st.metric("Hog ID", 
                             f"#{format_hog_id(least['id'])}", 
                             f"{least['adg']:.2f} kg/day")
                    # Ensure progress value is between 0 and 1
                    progress_value = max(0.0, min(1.0, (least['adg'] + 0.5) / 1.0))  # Shift and normalize
                    st.progress(progress_value, 
                              text=f"{least['adg']:.2f} kg/day")
        
            # --- Latest Week Summary ---
            st.markdown("---")
            st.header(f"📅 Weekly Performance ({summary_report['latest_week_date']})")
            
            if 'growth_status' in summary_report['latest_week']:
                st.info(summary_report['latest_week']['growth_status'])
            else:
                week_metrics = st.columns(3)
                
                with week_metrics[0]:
                    st.metric("Avg. Herd Gain", 
                             f"{summary_report['latest_week']['average_herd_gain']:.1f} kg",
                             help="Average weight gain across all hogs this week")
                
                if summary_report['latest_week']['highest_performer']:
                    hp = summary_report['latest_week']['highest_performer']
                    with week_metrics[1]:
                        st.metric("Top Gainer", 
                                 f"Hog #{hp['id']}", 
                                 f"+{hp['weight_change']:.1f} kg")
                
                if summary_report['latest_week']['least_performer']:
                    lp = summary_report['latest_week']['least_performer']
                    with week_metrics[2]:
                        st.metric("Lowest Gain", 
                                 f"Hog #{lp['id']}", 
                                 f"{lp['weight_change']:.1f} kg")
                
                # Simple Weekly Performance Distribution
                st.markdown("#### 📊 Weekly Weight Changes")
                if not all_growth_data_summary.empty and 'Week' in all_growth_data_summary.columns:
                    # Get latest week's data
                    latest_week = all_growth_data_summary['Week'].max()
                    week_data = all_growth_data_summary[all_growth_data_summary['Week'] == latest_week]
                    
                    if not week_data.empty:
                        # Simple bar chart showing each hog's weight change
                        fig = px.bar(
                            week_data.sort_values('Weight Change (kg)'),
                            x='Hog ID',
                            y='Weight Change (kg)',
                            color='Weight Change (kg)',
                            color_continuous_scale='RdYlGn',  # Red-Yellow-Green
                            title=f"Week {latest_week} - Weight Change by Hog"
                        )
                        
                        # Add a horizontal line at the average
                        avg_change = week_data['Weight Change (kg)'].mean()
                        fig.add_hline(
                            y=avg_change,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Average: {avg_change:.1f} kg",
                            annotation_position="bottom right"
                        )
                        
                        # Update layout for better readability
                        fig.update_layout(
                            xaxis_title="Hog ID",
                            yaxis_title="Weight Change (kg)",
                            showlegend=False,
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Simple statistics
                        stats = week_data['Weight Change (kg)'].describe()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Average Change", f"{stats['mean']:.1f} kg")
                            st.metric("Best Performer", 
                                     f"+{stats['max']:.1f} kg")
                        with col2:
                            st.metric("Number of Hogs", f"{len(week_data)}")
                            st.metric("Lowest Performer", 
                                     f"{stats['min']:.1f} kg")
                
                if summary_report['latest_week']['no_growth_hogs']:
                    no_growth_hogs = ", ".join([f"#{int(hid):03d}" for hid in summary_report['latest_week']['no_growth_hogs']])
                    st.warning(f"⚠️ **No growth this week:** {no_growth_hogs}")
            
            # --- Health Alerts ---
            if summary_report['potential_health_concern_hogs']:
                st.markdown("---")
                st.header("⚠️ Health Alerts")
                
                for concern in summary_report['potential_health_concern_hogs']:
                    st.error(
                        f"""
                        ### Hog #{concern['id']}
                        - **Issue:** {concern['consecutive_measurements']} consecutive non-positive weight changes
                        - **Last Checked:** {concern['latest_date']}
                        - **Action Required:** Immediate veterinary attention recommended
                        """
                    )
            
            # --- Recommendations ---
            st.markdown("---")
            st.header("📋 Action Items")
            
            if summary_report['potential_health_concern_hogs']:
                st.error(
                    """
                    ### ❗ Priority Actions
                    - Schedule veterinary check-ups for hogs with consecutive non-positive weight changes
                    - Isolate affected hogs to prevent potential spread of illness
                    - Review feeding and medication schedules
                    """
                )
            
            if summary_report['latest_week'].get('no_growth_hogs'):
                st.warning(
                    """
                    ### 🔍 Monitor Closely
                    - Check feeding patterns for hogs with no growth
                    - Verify water supply and quality
                    - Monitor for signs of stress or illness
                    - Consider environmental factors (temperature, space, etc.)
                    """
                )
            
            if not any([summary_report['potential_health_concern_hogs'], 
                       summary_report['latest_week'].get('no_growth_hogs')]):
                st.success(
                    """
                    ### ✅ All Systems Normal
                    - All hogs are showing normal growth patterns
                    - Continue with current feeding and care regimen
                    - Monitor for any changes in behavior or appetite
                    """
                )
            
            # --- Export Options ---
            st.markdown("---")
            st.header("📤 Export Report")
            
            col1, col2, _ = st.columns(3)
            
            with col1:
                # Create a simple text report
                report_text = generate_summary_report_text(summary_report)
                st.download_button(
                    label="📝 Download Text Report",
                    data=report_text,
                    file_name=f"hog_farm_report_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Option to generate a PDF (placeholder for future implementation)
                if st.button("📊 Generate PDF Report"):
                    st.info("PDF export coming soon! For now, please use the text report.")
                
            # Use the top-level generate_summary_report_text function


            # Suggestions based on observations
            st.subheader("Suggestions")
            suggestions = []

            if summary_report['latest_week'].get('no_growth_hogs'):
                suggestions.append("Investigate hogs with no growth. This could indicate dietary issues, stress, or health problems.")
            
            if summary_report.get('potential_health_concern_hogs'):
                suggestions.append("Prioritize hogs with potential health concerns for veterinary check-ups and specialized care.")

            if summary_report['latest_week'].get('least_performer'):
                suggestions.append("Review the diet and environment of the least performing hog(s). They might need specific adjustments.")

            if not suggestions:
                st.info("Current data indicates healthy growth patterns. Keep up the good work!")
            else:
                for suggestion in suggestions:
                    st.info(f"- {suggestion}")

    with tab4:
        # Only admin/staff can add/edit financials; viewers can only view
        can_edit_financials = st.session_state['user_role'] in ['admin', 'staff']
        st.header("💼 Financials")
        subtab_expenses, subtab_budgets, subtab_recurring, subtab_categories, subtab_insights, subtab_sales = st.tabs([
            "Expenses", "Budgets", "Recurring Expenses", "Categories", "Insights", "Sales"
        ])

        with subtab_expenses:
            # Add custom CSS for better styling
            st.markdown("""
                <style>
                .expense-card {
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    background-color: #f8f9fa;
                    border-left: 4px solid #4e79a7;
                    margin-bottom: 1.5rem;
                }
                .expense-form {
                    background-color: #ffffff;
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
                    margin-bottom: 1.5rem;
                }
                .expense-summary {
                    background-color: #f8f9fa;
                    border-radius: 0.5rem;
                    padding: 1rem;
                    margin-bottom: 1.5rem;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Page header with icon and description
            st.markdown("# 💰 Expenses")
            st.markdown("Track and manage all farm-related expenses in one place.")
            
            # Check permissions
            if not can_edit_financials:
                st.warning("🔒 You have view-only access to expenses. Contact an administrator for editing permissions.")
                # Show read-only view of expenses
                if not st.session_state['financial_data'].empty:
                    expenses_df = st.session_state['financial_data'][st.session_state['financial_data']['Type'] == 'Expense'].copy()
                    if not expenses_df.empty:
                        # Format the amount column and include subcategory in display
                        if 'Subcategory' in expenses_df.columns:
                            expenses_display = expenses_df[['Date', 'Category', 'Subcategory', 'Description', 'Amount']].copy()
                            # Combine subcategory into description if it exists
                            expenses_display['Description'] = expenses_display.apply(
                                lambda row: f"{row['Subcategory']}: {row['Description']}" 
                                if pd.notna(row['Subcategory']) and row['Subcategory'] != "" 
                                else row['Description'], 
                                axis=1
                            )
                            expenses_display = expenses_display[['Date', 'Category', 'Description', 'Amount']]
                        else:
                            expenses_display = expenses_df[['Date', 'Category', 'Description', 'Amount']].copy()
                        
                        expenses_display['Amount'] = expenses_display['Amount'].apply(lambda x: f"Kshs {x:,.2f}")
                        
                        # Filter the expenses based on the search criteria
                        filtered_expenses = expenses_display.copy()
                        if search_hog_id:
                            filtered_expenses = filtered_expenses[filtered_expenses['Description'].str.contains(search_hog_id, case=False)]
                        if search_start_date:
                            filtered_expenses = filtered_expenses[filtered_expenses['Date'] >= search_start_date]
                        if search_end_date:
                            filtered_expenses = filtered_expenses[filtered_expenses['Date'] <= search_end_date]
                        
                        # Display the filtered table
                        st.dataframe(
                            filtered_expenses,
                            column_config={
                                "Date": "Date",
                                "Category": "Category",
                                "Description": "Description",
                                "Amount": st.column_config.NumberColumn(
                                    "Amount",
                                    format="Kshs %.2f"
                                )
                            },
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("ℹ️ No expenses have been recorded yet.")
                else:
                    st.info("ℹ️ No financial data is currently available.")
            else:
                # --- Add New Expense Section ---
                with st.container():
                    st.markdown("### 📝 Add New Expense")
                
                # Get default categories and existing categories
                default_categories = ['Feed', 'Veterinary', 'Labor', 'Utilities', 'Equipment']
                existing_categories = []
                if not st.session_state['financial_data'].empty:
                    existing_categories = st.session_state['financial_data']['Category'].dropna().unique().tolist()
                    existing_categories = [cat for cat in existing_categories 
                                         if cat not in default_categories and cat != 'Other']
                
                # Get categories from session state if available
                persistent_categories = []
                if 'categories' in st.session_state:
                    persistent_categories = [
                        cat for cat in st.session_state['categories'].keys() 
                        if cat not in default_categories and cat != 'Other'
                    ]
                
                # Create category options
                category_options = ['Choose a Category'] + default_categories + existing_categories + persistent_categories + ['Other']
                seen = set()
                category_options = [x for x in category_options if not (x in seen or seen.add(x))]
                
                # Helper function to get category options
                def get_category_options():
                    options = ['Choose a Category']
                    options.extend(default_categories)
                    if 'categories' in st.session_state and st.session_state['categories']:
                        session_cats = [
                            cat for cat in st.session_state['categories'].keys() 
                            if cat not in default_categories and cat != 'Other'
                        ]
                        options.extend(session_cats)
                    options.append('Other')
                    return [x for x in options if x != '']
                
                # Category selection
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.session_state['expense_category'] = st.selectbox(
                        "Category:",
                        get_category_options(),
                        key='expense_category_interactive',
                        help="Select an existing category or choose 'Other' to create a new one"
                    )
                
                # New category input (shown only when 'Other' is selected)
                if st.session_state['expense_category'] == 'Other':
                    with col2:
                        st.session_state['new_category_any'] = st.text_input(
                            "New Category Name:",
                            key='new_category_any_interactive',
                            placeholder="Enter new category name"
                        )
                    st.session_state['new_subcategory'] = ''
                else:
                    st.session_state['new_category_any'] = ''
                    
                    # Initialize subcategory in session state if not exists
                    if 'new_subcategory' not in st.session_state:
                        st.session_state['new_subcategory'] = ""
                    
                    # Subcategory selection (only shown when a valid category is selected)
                    if st.session_state['expense_category'] != 'Choose a Category':
                        # Get subcategories for the selected category
                        subcategories = []
                        if 'categories' in st.session_state and st.session_state['categories']:
                            category = st.session_state['expense_category']
                            if category in st.session_state['categories']:
                                # Extract subcategory names if they are dictionaries
                                raw_subcategories = st.session_state['categories'][category].get('subcategories', [])
                                subcategories = [
                                    subcat['name'] if isinstance(subcat, dict) and 'name' in subcat 
                                    else str(subcat) 
                                    for subcat in raw_subcategories
                                ]
                        
                        # Add subcategory selection
                        subcategory_col1, subcategory_col2 = st.columns([2, 1])
                        with subcategory_col1:
                            # Use a unique key for the selectbox
                            selected_subcategory = st.selectbox(
                                "Subcategory:",
                                ["Select a subcategory..."] + subcategories,
                                key=f"subcategory_{st.session_state['expense_category']}",
                                help="Select a subcategory or enter a new one below"
                            )
                            
                            # Update the session state when selection changes
                            if selected_subcategory != "Select a subcategory...":
                                st.session_state['new_subcategory'] = selected_subcategory
                        
                        # Option to add new subcategory
                        if selected_subcategory == "Select a subcategory...":
                            with subcategory_col2:
                                new_subcategory = st.text_input(
                                    "New Subcategory:",
                                    key=f"new_subcategory_{st.session_state['expense_category']}",
                                    placeholder="Enter new subcategory"
                                )
                                
                                # Add button to save new subcategory
                                if st.button("Add Subcategory"):
                                    if new_subcategory.strip():
                                        if 'categories' not in st.session_state:
                                            st.session_state['categories'] = {}
                                        if st.session_state['expense_category'] not in st.session_state['categories']:
                                            st.session_state['categories'][st.session_state['expense_category']] = {'subcategories': [], 'color': '#CCCCCC'}
                                        
                                        if 'subcategories' not in st.session_state['categories'][st.session_state['expense_category']]:
                                            st.session_state['categories'][st.session_state['expense_category']]['subcategories'] = []
                                        
                                        # Get the current subcategories
                                        current_subs = st.session_state['categories'][st.session_state['expense_category']]['subcategories']
                                        
                                        # Check if subcategory already exists (handling both dict and string formats)
                                        sub_exists = any(
                                            (isinstance(sub, dict) and sub.get('name') == new_subcategory.strip()) or 
                                            (isinstance(sub, str) and sub == new_subcategory.strip())
                                            for sub in current_subs
                                        )
                                        
                                        if not sub_exists:
                                            # Add as a simple string to maintain consistency
                                            current_subs.append(new_subcategory.strip())
                                            st.success(f"Added subcategory: {new_subcategory.strip()}")
                                            st.rerun()
                                        else:
                                            st.warning("This subcategory already exists.")
                                            st.session_state['new_subcategory'] = new_subcategory.strip()
                                    else:
                                        st.warning("Please enter a subcategory name.")
                
                # Initialize form key if not exists
                if 'form_key' not in st.session_state:
                    st.session_state['form_key'] = 0
                
                # Create a form with a unique key that changes on submission
                form_key = f"expense_form_{st.session_state['form_key']}"
                with st.form(key=form_key):
                    expense_col1, expense_col2 = st.columns(2)
                    
                    with expense_col1:
                        expense_date = st.date_input(
                            "Date",
                            value=dt.date.today(),
                            key=f"{form_key}_date",
                            help="Select the date of the expense"
                        )
                        
                    with expense_col2:
                        expense_amount = st.number_input(
                            "Amount (Kshs)",
                            min_value=0.0,
                            step=100.0,
                            format="%.2f",
                            key=f"{form_key}_amount",
                            help="Enter the expense amount in Kshs"
                        )
                    
                    expense_description = st.text_area(
                        "Description",
                        key=f"{form_key}_description",
                        placeholder="Enter a brief description of the expense",
                        max_chars=200,
                        height=80
                    )
                    
                    # Get the current values from the form
                    current_amount = expense_amount
                    current_description = expense_description
                    
                    submit_col1, submit_col2 = st.columns([1, 4])
                    with submit_col1:
                        submit_expense = st.form_submit_button(
                            "➕ Add Expense",
                            type="primary",
                            use_container_width=True
                        )
                    
                    if submit_expense:
                        # Validate form
                        expense_category = st.session_state.get('expense_category', 'Choose a Category')
                        new_category = st.session_state.get('new_category_any', '')
                        
                        if expense_category == 'Choose a Category':
                            st.error('❌ Please select a valid category.')
                            st.stop()
                        
                        # Handle 'Other' category
                        final_category = expense_category
                        if expense_category == 'Other':
                            if new_category and new_category.strip():
                                final_category = new_category.strip()
                                # Initialize categories if not exists
                                if 'categories' not in st.session_state:
                                    st.session_state['categories'] = {
                                        'Feed': {'color': '#FFD700', 'subcategories': ['Starter', 'Grower', 'Finisher', 'Other']},
                                        'Veterinary': {'color': '#FF6347', 'subcategories': ['Medication', 'Consultation', 'Vaccination', 'Other']},
                                        'Labor': {'color': '#87CEEB', 'subcategories': ['Wages', 'Casual', 'Other']},
                                        'Utilities': {'color': '#90EE90', 'subcategories': ['Water', 'Electricity', 'Other']},
                                        'Equipment': {'color': '#DDA0DD', 'subcategories': ['Repair', 'Purchase', 'Other']}
                                    }
                            else:
                                st.error("❌ Please enter a name for the new category.")
                                st.stop()
                        
                        if not current_description.strip() or current_amount <= 0:
                            st.error("❌ Please fill in all required fields and ensure amount is greater than 0.")
                            st.stop()
                        
                        # Get the subcategory if available
                        subcategory = st.session_state.get('new_subcategory', '')
                        
                        # If we have a subcategory, use it in the description
                        description = f"{subcategory}: {current_description}" if subcategory and subcategory != "Select a subcategory..." else current_description
                        
                        # Add the expense
                        try:
                            new_expense = pd.DataFrame([{
                                'Date': expense_date,
                                'Type': 'Expense',
                                'Category': final_category,
                                'Subcategory': subcategory if subcategory and subcategory != "Select a subcategory..." else "",
                                'Description': description,
                                'Hog ID': pd.NA,
                                'Weight (kg)': pd.NA,
                                'Price/kg': pd.NA,
                                'Amount': current_amount
                            }])
                            
                            st.session_state['financial_data'] = pd.concat(
                                [st.session_state['financial_data'], new_expense], 
                                ignore_index=True
                            )
                            save_financial_data(st.session_state['financial_data'])
                            st.success("✅ Expense added successfully!")
                            st.balloons()
                            
                            # Increment form key to force a new form instance
                            st.session_state['form_key'] += 1
                            
                            # Reset the category and subcategory
                            st.session_state['expense_category'] = 'Choose a Category'
                            st.session_state['new_subcategory'] = ""
                            
                            # Force a rerun to update the UI with default values
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ An error occurred while saving the expense: {str(e)}")
            
            # Divider before the next section
            # --- Expense Analytics Section ---
            if not st.session_state['financial_data'].empty:
                expenses_df = st.session_state['financial_data'][st.session_state['financial_data']['Type'] == 'Expense'].copy()
                
                if not expenses_df.empty:
                    # Convert date to datetime if it's not already
                    expenses_df['Date'] = pd.to_datetime(expenses_df['Date'])
                    
                    # --- Summary Cards ---
                    st.markdown("### 📊 Expense Summary")
                    
                    # Calculate summary metrics
                    total_expenses = expenses_df['Amount'].sum()
                    avg_expense = expenses_df['Amount'].mean()
                    expense_count = len(expenses_df)
                    
                    # Create summary cards
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Total Expenses",
                            value=f"Kshs {total_expenses:,.2f}",
                            help="Sum of all recorded expenses"
                        )
                    
                    with col2:
                        st.metric(
                            label="Average Expense",
                            value=f"Kshs {avg_expense:,.2f}",
                            help="Average amount per expense"
                        )
                    
                    with col3:
                        st.metric(
                            label="Total Records",
                            value=f"{expense_count:,}",
                            help="Number of expense records"
                        )
                    
                    # --- Expense Category Breakdown ---
                    st.markdown("### 📈 Expense Analysis")
                    
                    # Create tabs for different visualizations
                    tab1, tab2 = st.tabs(["By Category", "Over Time"])
                    
                    with tab1:
                        st.markdown("#### 🥧 Expense Distribution by Category")
                        category_summary = expenses_df.groupby('Category')['Amount'].agg(['sum', 'count']).reset_index()
                        category_summary.columns = ['Category', 'Total Amount', 'Number of Expenses']
                        category_summary = category_summary.sort_values('Total Amount', ascending=False)
                        
                        # Display top categories as metrics
                        if len(category_summary) > 0:
                            top_categories = category_summary.head(3)
                            cols = st.columns(3)
                            for idx, (_, row) in enumerate(top_categories.iterrows()):
                                with cols[idx % 3]:
                                    percentage = (row['Total Amount'] / total_expenses) * 100
                                    st.metric(
                                        label=f"{row['Category']}",
                                        value=f"Kshs {row['Total Amount']:,.2f}",
                                        delta=f"{percentage:.1f}% of total"
                                    )
                        
                        # Show full category breakdown in columns
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown("#### 📋 Category Details")
                            st.dataframe(
                                category_summary.rename(columns={
                                    'Category': 'Category',
                                    'Total Amount': 'Total (Kshs)',
                                    'Number of Expenses': 'Count'
                                }).style.format({
                                    'Total (Kshs)': '{:,.2f}'
                                }),
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        with col2:
                            st.markdown("#### 📊 Distribution")
                            fig_pie = px.pie(
                                category_summary,
                                values='Total Amount',
                                names='Category',
                                color_discrete_sequence=px.colors.qualitative.Pastel,
                                hole=0.3
                            )
                            fig_pie.update_traces(
                                textposition='inside',
                                textinfo='percent+label',
                                hovertemplate='%{label}<br>Kshs %{value:,.2f}<br>%{percent}'
                            )
                            fig_pie.update_layout(
                                showlegend=False,
                                margin=dict(t=0, b=0, l=0, r=0)
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with tab2:
                        st.markdown("#### 📅 Monthly Expense Trends")
                        
                        # Group by month and category
                        monthly_expenses = expenses_df.copy()
                        monthly_expenses['Month'] = monthly_expenses['Date'].dt.to_period('M').astype(str)
                        monthly_summary = monthly_expenses.groupby(['Month', 'Category'])['Amount'].sum().reset_index()
                        
                        # Create line chart
                        fig_line = px.line(
                            monthly_summary,
                            x='Month',
                            y='Amount',
                            color='Category',
                            title="Monthly Expenses by Category",
                            markers=True,
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig_line.update_layout(
                            xaxis_title="Month",
                            yaxis_title="Amount (Kshs)",
                            legend_title="Category",
                            hovermode='x unified'
                        )
                        fig_line.update_traces(
                            hovertemplate='%{x}<br>Kshs %{y:,.2f}<extra></extra>'
                        )
                        st.plotly_chart(fig_line, use_container_width=True)
                        
                        # Monthly summary table
                        st.markdown("#### 📅 Monthly Summary")
                        monthly_totals = monthly_expenses.groupby('Month')['Amount'].agg(['sum', 'count']).reset_index()
                        monthly_totals.columns = ['Month', 'Total Amount', 'Number of Expenses']
                        monthly_totals = monthly_totals.sort_values('Month')
                        
                        # Calculate month-over-month change
                        monthly_totals['MoM Change'] = monthly_totals['Total Amount'].pct_change() * 100
                        
                        st.dataframe(
                            monthly_totals.style.format({
                                'Total Amount': 'Kshs {:,.2f}',
                                'MoM Change': '{:,.1f}%'
                            }).apply(
                                lambda x: [''] * len(x) if x.name != len(monthly_totals)-1 
                                else [''] + [''] + [
                                    'color: green' if val > 0 else 'color: red' if val < 0 else ''
                                    for val in x[2:]
                                ],
                                axis=1
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    # --- All Expenses Table ---
                    # Add a toggle to show/hide the expenses table
                    show_expenses = st.checkbox(
                        "📋 Show All Expenses",
                        value=False,
                        key='show_all_expenses',
                        help="Toggle to show/hide the detailed expenses table"
                    )
                    
                    if show_expenses:
                        st.markdown("### 📋 All Expenses")
                        
                        # Format the display dataframe
                        expenses_display = expenses_df[['Date', 'Category', 'Description', 'Amount']].copy()
                        expenses_display['Date'] = expenses_display['Date'].dt.strftime('%Y-%m-%d')
                        
                        # Add Excel export button
                        if not expenses_df.empty:
                            # Create a buffer to store the Excel file
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                # Export the actual data (not the display version)
                                expenses_df.to_excel(writer, index=False, sheet_name='Expenses')
                                
                                # Get the xlsxwriter workbook and worksheet objects
                                workbook = writer.book
                                worksheet = writer.sheets['Expenses']
                                
                                # Add a header format
                                header_format = workbook.add_format({
                                    'bold': True,
                                    'text_wrap': True,
                                    'valign': 'top',
                                    'fg_color': '#4CAF50',
                                    'border': 1,
                                    'font_color': 'white'
                                })
                                
                                # Write the column headers with the defined format
                                for col_num, value in enumerate(expenses_df.columns.values):
                                    worksheet.write(0, col_num, value, header_format)
                                
                                # Set column widths
                                worksheet.set_column('A:A', 15)  # Date
                                worksheet.set_column('B:B', 20)  # Category
                                worksheet.set_column('C:C', 15)  # Subcategory if exists
                                worksheet.set_column('D:D', 40)  # Description
                                worksheet.set_column('E:E', 15)  # Amount
                                
                                # Format the amount column
                                money_format = workbook.add_format({'num_format': 'Kshs #,##0.00'})
                                worksheet.set_column('E:E', 15, money_format)
                            
                            # Create export buttons in a row
                            col1, col2, _ = st.columns([1, 1, 3])
                            
                            with col1:
                                # Excel Export Button
                                st.download_button(
                                    label="📊 Excel",
                                    data=output.getvalue(),
                                    file_name=f"expenses_export_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    help="Download all expenses as an Excel file",
                                    use_container_width=True
                                )
                            
                            with col2:
                                # CSV Export Button
                                csv = expenses_df.to_csv(index=False)
                                st.download_button(
                                    label="📝 CSV",
                                    data=csv,
                                    file_name=f"expenses_export_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    help="Download all expenses as a CSV file",
                                    use_container_width=True
                                )
                    
                        # Add search and filter functionality
                        search_col, month_col, category_col = st.columns([3, 1, 1])
                        
                        with search_col:
                            search_term = st.text_input("Search expenses", "", 
                                                      placeholder="Search by description or category...")
                        
                        with month_col:
                            # Add month filter
                            month_list = ['All Months'] + sorted(
                                pd.to_datetime(expenses_df['Date']).dt.strftime('%Y-%m').unique().tolist(), 
                                reverse=True
                            )
                            month_filter = st.selectbox(
                                "Filter by month",
                                month_list,
                                key="expense_month_filter"
                            )
                        
                        with category_col:
                            # Add category filter
                            category_filter = st.selectbox(
                                "Filter by category",
                                ["All Categories"] + sorted(expenses_df['Category'].unique().tolist()),
                                key="expense_category_filter"
                            )
                        
                        # Create a filtered copy of the original data
                        filtered_expenses = expenses_df.copy()
                        
                        # Apply search filter
                        if search_term:
                            search_mask = (filtered_expenses['Description'].str.contains(search_term, case=False, na=False)) | \
                                         (filtered_expenses['Category'].str.contains(search_term, case=False, na=False))
                            filtered_expenses = filtered_expenses[search_mask]
                        
                        # Apply month filter
                        if month_filter != "All Months":
                            filtered_expenses = filtered_expenses[
                                pd.to_datetime(filtered_expenses['Date']).dt.strftime('%Y-%m') == month_filter
                            ]
                        
                        # Apply category filter
                        if category_filter != "All Categories":
                            filtered_expenses = filtered_expenses[filtered_expenses['Category'] == category_filter]
                        
                        # Create display version with formatted amounts
                        if not filtered_expenses.empty:
                            display_expenses = filtered_expenses[['Date', 'Category', 'Description', 'Amount']].copy()
                            display_expenses['Date'] = pd.to_datetime(display_expenses['Date']).dt.strftime('%Y-%m-%d')
                            
                            # Display the table with custom styling
                            st.dataframe(
                                display_expenses,
                                column_config={
                                    "Date": "Date",
                                    "Category": "Category",
                                    "Description": "Description",
                                    "Amount": st.column_config.NumberColumn(
                                        "Amount (Kshs)",
                                        format="%.2f"
                                    )
                                },
                                use_container_width=True,
                                hide_index=True,
                                height=400
                            )
                            
                            # Add export buttons
                            export_col1, export_col2 = st.columns(2)
                            
                            with export_col1:
                                # Export to Excel
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    filtered_expenses.to_excel(writer, index=False, sheet_name='Expenses')
                                    workbook = writer.book
                                    worksheet = writer.sheets['Expenses']
                                    
                                    # Format headers
                                    header_format = workbook.add_format({
                                        'bold': True,
                                        'text_wrap': True,
                                        'valign': 'top',
                                        'fg_color': '#4CAF50',
                                        'border': 1,
                                        'font_color': 'white'
                                    })
                                    
                                    for col_num, value in enumerate(filtered_expenses.columns.values):
                                        worksheet.write(0, col_num, value, header_format)
                                    
                                    # Set column widths
                                    worksheet.set_column('A:A', 15)  # Date
                                    worksheet.set_column('B:B', 20)  # Category
                                    worksheet.set_column('C:C', 15)  # Subcategory
                                    worksheet.set_column('D:D', 40)  # Description
                                    worksheet.set_column('E:E', 15, workbook.add_format({'num_format': 'Kshs #,##0.00'}))  # Amount
                                
                                st.download_button(
                                    label="📊 Export to Excel",
                                    data=output.getvalue(),
                                    file_name=f"expenses_export_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            
                            with export_col2:
                                # Export to CSV
                                csv = filtered_expenses.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📝 Export to CSV",
                                    data=csv,
                                    file_name=f"expenses_export_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime='text/csv',
                                    use_container_width=True
                                )
                        else:
                            st.info("No expenses match the current filters.")
                        
                        # --- Expense Management ---
                        st.markdown("### ⚙️ Expense Management")
                    
                        # Create a dropdown for selecting an expense to edit/delete
                        expenses_df['Label'] = expenses_df.apply(
                            lambda row: f"{row['Date'].strftime('%Y-%m-%d')} | {row['Category']} | {row['Description']} | Kshs {row['Amount']:,.2f}", 
                            axis=1
                        )
                        
                        # Use a session state to track if we should reset the selectbox
                        if 'reset_expense_select' not in st.session_state:
                            st.session_state.reset_expense_select = False
                        
                        # Create a unique key for the selectbox
                        selectbox_key = f"expense_select_{len(expenses_df)}"
                        
                        # Create the selectbox
                        selected_label = st.selectbox(
                            "Select an expense to manage:", 
                            ["Select an expense..."] + expenses_df['Label'].tolist(), 
                            key=selectbox_key
                        )
                        
                        # Check if we need to show the expense form
                        show_expense_form = selected_label and selected_label != "Select an expense..."
                        
                        if show_expense_form and not st.session_state.reset_expense_select:
                            matching_expenses = expenses_df[expenses_df['Label'] == selected_label]
                            if not matching_expenses.empty:
                                idx = matching_expenses.index[0]
                                action_key = f'action_expense_{idx}'
                                
                                # Store the selected action in session state
                                if action_key not in st.session_state:
                                    st.session_state[action_key] = "Edit"  # Default to Edit
                                
                                # Use a radio button to select action
                                action = st.radio(
                                    "Action:", 
                                    ["Edit", "Delete"], 
                                    key=f'radio_{action_key}'
                                )
                                
                                # Update the session state when radio changes
                                if action != st.session_state[action_key]:
                                    st.session_state[action_key] = action
                                    st.rerun()
                                
                                if action == "Edit":
                                    with st.form(key=f'edit_expense_form_{idx}'):
                                        st.markdown("---")
                                        st.subheader(f"Edit Expense #{idx+1}")
                                        edit_date = st.date_input("Date:", value=pd.to_datetime(expenses_df.loc[idx, 'Date']).date(), key=f'edit_expense_date_{idx}')
                                        edit_category = st.selectbox("Category:", ['Feed', 'Veterinary', 'Labor', 'Utilities', 'Equipment', 'Other'], 
                                                                  index=['Feed', 'Veterinary', 'Labor', 'Utilities', 'Equipment', 'Other'].index(expenses_df.loc[idx, 'Category']) 
                                                                  if expenses_df.loc[idx, 'Category'] in ['Feed', 'Veterinary', 'Labor', 'Utilities', 'Equipment', 'Other'] else 0, 
                                                                  key=f'edit_expense_category_{idx}')
                                        edit_description = st.text_input("Description:", value=expenses_df.loc[idx, 'Description'], key=f'edit_expense_description_{idx}')
                                        edit_amount = st.number_input("Amount:", min_value=0.0, value=float(expenses_df.loc[idx, 'Amount']), format="%.2f", key=f'edit_expense_amount_{idx}')
                                        
                                        col1, col2 = st.columns([1, 1])
                                        with col1:
                                            submit_edit = st.form_submit_button("💾 Save Changes")
                                        with col2:
                                            cancel_edit = st.form_submit_button("❌ Cancel")
                                        
                                        if submit_edit:
                                            st.session_state['financial_data'].at[expenses_df.index[idx], 'Date'] = edit_date
                                            st.session_state['financial_data'].at[expenses_df.index[idx], 'Category'] = edit_category
                                            st.session_state['financial_data'].at[expenses_df.index[idx], 'Description'] = edit_description
                                            st.session_state['financial_data'].at[expenses_df.index[idx], 'Amount'] = edit_amount
                                            save_financial_data(st.session_state['financial_data'])
                                            st.success("Expense updated successfully!")
                                            st.balloons()
                                            st.session_state.reset_expense_select = True
                                            st.rerun()
                                            
                                        if cancel_edit:
                                            st.session_state.reset_expense_select = True
                                            st.rerun()
                                
                                elif action == "Delete":
                                    st.markdown("---")
                                    st.subheader(f"Delete Expense #{idx+1}")
                                    st.warning("⚠️ Are you sure you want to delete this expense? This action cannot be undone.")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        if st.button("✅ Confirm Delete", type="primary", key=f'confirm_delete_{idx}'):
                                            # Remove the expense from the financial data
                                            st.session_state['financial_data'] = st.session_state['financial_data'].drop(index=idx).reset_index(drop=True)
                                            
                                            # Save the changes
                                            save_financial_data(st.session_state['financial_data'])
                                            st.success("Expense deleted successfully!")
                                            st.balloons()
                                            st.session_state.reset_expense_select = True
                                            st.rerun()
                                    
                                    with col2:
                                        if st.button("❌ Cancel", key=f'cancel_delete_{idx}'):
                                            st.session_state.reset_expense_select = True
                                            st.rerun()
                        
                        # Reset the flag after handling the rerun
                        if st.session_state.reset_expense_select:
                            st.session_state.reset_expense_select = False
                            st.rerun()
                else:
                    st.info("No expenses recorded yet.")
            else:
                st.info("No financial data available yet.")

        with subtab_budgets:
            if not can_edit_financials:
                st.info("You do not have permission to manage budgets. Viewing only.")
                if 'budgets' not in st.session_state:
                    st.session_state['budgets'] = pd.DataFrame(columns=['Category', 'Month', 'Budget'])
                budgets_df = st.session_state['budgets']
                if not budgets_df.empty:
                    st.dataframe(budgets_df, use_container_width=True)
                else:
                    st.info("No budgets to display.")
            else:
                # --- Budget Management Section ---
                st.subheader("💸 Budget Management")
            if 'budgets' not in st.session_state:
                st.session_state['budgets'] = pd.DataFrame(columns=['Category', 'Month', 'Budget'])
            budgets_df = st.session_state['budgets']
            with st.expander("Set Monthly Budgets per Category"):
                categories = st.session_state['financial_data']['Category'].dropna().unique().tolist() if not st.session_state['financial_data'].empty else []
                categories = sorted(list(set(categories + ['Feed', 'Veterinary', 'Labor', 'Utilities', 'Equipment', 'Other'])))
                month = st.selectbox("Select Month:", pd.date_range(start=pd.Timestamp('today').replace(day=1)-pd.DateOffset(months=6), periods=12, freq='MS').strftime('%Y-%m').tolist(), key='budget_month')
                category = st.selectbox("Select Category:", categories, key='budget_category')
                budget_amount = st.number_input("Set Budget (Kshs):", min_value=0.0, step=100.0, format="%.2f", key='budget_amount')
                if st.button("Save Budget", key='save_budget_btn'):
                    budgets_df = budgets_df[(budgets_df['Category'] != category) | (budgets_df['Month'] != month)]
                    budgets_df = pd.concat([budgets_df, pd.DataFrame([{ 'Category': category, 'Month': month, 'Budget': budget_amount }])], ignore_index=True)
                    st.session_state['budgets'] = budgets_df
                    st.success(f"Budget for {category} ({month}) saved!")
                    st.rerun()
            with st.expander("Edit or Delete Existing Budgets"):
                if not budgets_df.empty:
                    budgets_df = budgets_df.reset_index(drop=True)
                    budgets_df['Label'] = budgets_df['Category'] + ' (' + budgets_df['Month'] + ')'
                    selected = st.selectbox("Select Budget to Edit/Delete:", budgets_df['Label'].tolist(), key='edit_budget_select')
                    if selected:
                        idx = budgets_df[budgets_df['Label'] == selected].index[0]
                        edit_amount = st.number_input("Edit Budget Amount (Kshs):", min_value=0.0, value=float(budgets_df.loc[idx, 'Budget']), step=100.0, format="%.2f", key='edit_budget_amount')
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Save Changes", key='save_edit_budget_btn'):
                                budgets_df.at[idx, 'Budget'] = edit_amount
                                st.session_state['budgets'] = budgets_df.drop(columns=['Label'])
                                st.success("Budget updated!")
                                st.rerun()
                        with col2:
                            if st.button("Delete Budget", key='delete_budget_btn'):
                                budgets_df = budgets_df.drop(idx).drop(columns=['Label'])
                                st.session_state['budgets'] = budgets_df
                                st.success("Budget deleted!")
                                st.rerun()
                else:
                    st.info("No budgets to edit or delete.")

        with subtab_recurring:
            if not can_edit_financials:
                st.info("You do not have permission to manage recurring expenses.")
            else:
                # --- Recurring Expenses Section ---
                st.subheader("🔁 Recurring Expenses")
            if 'recurring_expenses' not in st.session_state:
                st.session_state['recurring_expenses'] = []
            with st.expander("Set Up Recurring Expense"):
                # Function to get all available categories
                def get_all_categories():
                    # Start with default categories
                    default_categories = ['Feed', 'Veterinary', 'Labor', 'Utilities', 'Equipment']
                    
                    # Start with default options
                    options = ['Choose a Category']
                    
                    # Add default categories first
                    options.extend(default_categories)
                    
                    # Add categories from session state if available
                    if 'categories' in st.session_state and st.session_state['categories']:
                        # Get all categories except those already in default_categories
                        session_cats = [
                            cat for cat in st.session_state['categories'].keys() 
                            if cat not in default_categories and cat != 'Other'
                        ]
                        options.extend(session_cats)
                    
                    # Add 'Other' at the end
                    options.append('Other')
                    
                    # Remove duplicates and empty strings while preserving order
                    seen = set()
                    return [x for x in options if not (x in seen or seen.add(x)) and x != '']
                
                # Get current category options
                category_options = get_all_categories()
                
                # Display the selectbox
                rec_category = st.selectbox("Category:", category_options, key='rec_category')
                rec_amount = st.number_input("Amount (Kshs):", min_value=0.0, step=100.0, format="%.2f", key='rec_amount')
                rec_frequency = st.selectbox("Frequency:", ['Monthly', 'Quarterly'], key='rec_frequency')
                rec_start_date = st.date_input("Start Date:", value=dt.date.today(), key='rec_start_date')
                rec_description = st.text_input("Description (optional):", key='rec_description')
                if st.button("Add Recurring Expense", key='add_rec_expense_btn'):
                    st.session_state['recurring_expenses'].append({
                        'Category': rec_category,
                        'Amount': rec_amount,
                        'Frequency': rec_frequency,
                        'Start Date': rec_start_date,
                        'Description': rec_description
                    })
                    st.success("Recurring expense added!")
                    st.rerun()
            if st.session_state['recurring_expenses']:
                st.markdown("**Your Recurring Expenses:**")
                for idx, rec in enumerate(st.session_state['recurring_expenses']):
                    st.write(f"{idx+1}. {rec['Category']} - {rec['Amount']:.2f} Kshs, {rec['Frequency']}, Start: {rec['Start Date']}, Desc: {rec['Description']}")
            else:
                st.info("No recurring expenses set.")

        with subtab_categories:
            if not can_edit_financials:
                st.info("🔒 You don't have permission to manage categories. Please contact an administrator.")
            else:
                # --- Category Management Section ---
                st.markdown("""
            <style>
            .category-card {
                padding: 1.5rem;
                border-radius: 0.5rem;
                background-color: #f8f9fa;
                border-left: 4px solid #4e79a7;
                margin-bottom: 1.5rem;
            }
            .subcategory-item {
                padding: 0.5rem;
                margin: 0.25rem 0;
                border-radius: 0.25rem;
                background-color: #ffffff;
                border: 1px solid #e9ecef;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            # 🗂️ Category Management
            Organize your expense categories and subcategories with custom colors for better tracking and reporting.
            """)
            
            # Default categories with improved color scheme
            default_categories = {
                'Feed': {
                    'color': '#4E79A7',  # Soft blue
                    'subcategories': [
                        {'name': 'Starter', 'color': '#5D8BCD'},
                        {'name': 'Grower', 'color': '#7FA6D9'},
                        {'name': 'Finisher', 'color': '#A1C1E5'},
                        {'name': 'Supplements', 'color': '#C3DCF1'}
                    ]
                },
                'Veterinary': {
                    'color': '#F28E2B',  # Warm orange
                    'subcategories': [
                        {'name': 'Medication', 'color': '#F4A760'},
                        {'name': 'Checkup', 'color': '#F7C08E'},
                        {'name': 'Vaccination', 'color': '#F9D9BC'},
                        {'name': 'Emergency', 'color': '#FCF2E9'}
                    ]
                },
                'Labor': {
                    'color': '#E15759',  # Soft red
                    'subcategories': [
                        {'name': 'Wages', 'color': '#E77C73'},
                        {'name': 'Overtime', 'color': '#ECA19B'},
                        {'name': 'Contract', 'color': '#F1C6C3'}
                    ]
                },
                'Utilities': {
                    'color': '#76B7B2',  # Teal
                    'subcategories': [
                        {'name': 'Water', 'color': '#8FC5C1'},
                        {'name': 'Electricity', 'color': '#A7D3D0'},
                        {'name': 'Internet', 'color': '#BFE1E0'}
                    ]
                },
                'Equipment': {
                    'color': '#59A14F',  # Green
                    'subcategories': [
                        {'name': 'Maintenance', 'color': '#7AB874'},
                        {'name': 'Purchase', 'color': '#9BCF95'},
                        {'name': 'Repair', 'color': '#BCE7B7'}
                    ]
                },
                'Other': {
                    'color': '#B07AA1',  # Muted purple
                    'subcategories': [
                        {'name': 'Miscellaneous', 'color': '#C397B9'},
                        {'name': 'Unexpected', 'color': '#D6B5D0'}
                    ]
                }
            }
            
            # Initialize categories in session state if not present
            if 'categories' not in st.session_state:
                st.session_state['categories'] = default_categories.copy()
            
            # Get existing categories from financial data
            existing_categories = []
            if not st.session_state['financial_data'].empty:
                existing_categories = st.session_state['financial_data']['Category'].dropna().unique().tolist()
                # Filter out default categories and 'Other'
                existing_categories = [cat for cat in existing_categories 
                                     if cat not in default_categories and cat != 'Other']
            
            # Helper function to check if two colors are similar
            def colors_are_similar(hex1, hex2, threshold=100):
                # Convert hex to RGB
                def hex_to_rgb(hex_color):
                    hex_color = hex_color.lstrip('#')
                    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                
                rgb1 = hex_to_rgb(hex1)
                rgb2 = hex_to_rgb(hex2)
                
                # Calculate Euclidean distance in RGB space
                distance = sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5
                return distance < threshold
            
            # Function to generate a visually distinct color
            def get_distinct_color():
                # Start with a set of visually distinct colors
                distinct_colors = [
                    '#FFD700', '#FF6347', '#87CEEB', '#90EE90', '#DDA0DD',
                    '#FFA07A', '#20B2AA', '#9370DB', '#FF69B4', '#32CD32',
                    '#FF8C00', '#9932CC', '#00BFFF', '#FF4500', '#8A2BE2',
                    '#1E90FF', '#FF1493', '#00FA9A', '#8B008B', '#FFD700'
                ]
                
                # Get all existing colors
                existing_colors = [cat['color'] for cat in st.session_state['categories'].values()]
                
                # First, try to use one of the predefined distinct colors
                for color in distinct_colors:
                    if color not in existing_colors:
                        return color
                
                # If all distinct colors are used, generate a random color
                while True:
                    # Generate a random color in HSV space for better distribution
                    h = random.random()
                    s = 0.7 + random.random() * 0.3  # Saturation between 0.7-1.0
                    v = 0.7 + random.random() * 0.3  # Value between 0.7-1.0
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
                    
                    # Ensure the color is not too similar to existing colors
                    if all(not colors_are_similar(color, existing) for existing in existing_colors):
                        return color
            # Ensure all existing categories from financial data are in our categories
            for cat in existing_categories:
                if cat not in st.session_state['categories']:
                    new_color = get_distinct_color()
                    st.session_state['categories'][cat] = {
                        'color': new_color,
                        'subcategories': ['Other']
                    }
            
            categories_dict = st.session_state['categories']
            
            # Create tabs for better organization
            tab_add, tab_manage = st.tabs(["➕ Add New", "✏️ Manage"])
            
            with tab_add:
                st.markdown("### ➕ Add New Category")
                with st.form(key='add_category_form'):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        new_cat = st.text_input("Category Name", 
                                              placeholder="e.g., Transportation, Supplies",
                                              key='new_cat')
                    with col2:
                        st.markdown("<div style='height: 30px; display: flex; align-items: flex-end;'>"
                                  "<div>Color</div></div>", 
                                  unsafe_allow_html=True)
                        cat_color = st.color_picker("", 
                                                  value=get_distinct_color(),
                                                  key='new_cat_color',
                                                  label_visibility="collapsed")
                    
                    if st.form_submit_button("Add Category", use_container_width=True):
                        if new_cat and new_cat.strip():
                            if new_cat not in categories_dict:
                                categories_dict[new_cat] = {
                                    'color': cat_color, 
                                    'subcategories': [{'name': 'Other', 'color': '#D3D3D3'}]
                                }
                                st.session_state['categories'] = categories_dict
                                st.success(f"✅ Category '{new_cat}' added successfully!")
                                st.rerun()
                            else:
                                st.error("A category with this name already exists.")
                        else:
                            st.warning("Please enter a category name")

                st.markdown("---")
                st.markdown("### ➕ Add New Subcategory")
                
                if not categories_dict:
                    st.info("No categories available. Please add a category first.")
                else:
                    selected_category = st.selectbox("Select Category", 
                                                  ["Select a category..."] + sorted(list(categories_dict.keys())),
                                                  key='cat_for_sub')
                    
                    if selected_category and selected_category != "Select a category...":
                        with st.form(key='add_subcategory_form'):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                new_subcat = st.text_input("Subcategory Name",
                                                         placeholder="e.g., Maintenance, Supplies",
                                                         key='new_subcat')
                            with col2:
                                st.markdown("<div style='height: 30px; display: flex; align-items: flex-end;'>"
                                          "<div>Color</div></div>", 
                                          unsafe_allow_html=True)
                                subcat_color = st.color_picker("", 
                                                             value=get_distinct_color(),
                                                             key='new_subcat_color',
                                                             label_visibility="collapsed")
                            
                            if st.form_submit_button("Add Subcategory", use_container_width=True):
                                if new_subcat and new_subcat.strip():
                                    subcategories = categories_dict[selected_category]['subcategories']
                                    subcategory_names = [
                                        s['name'].lower() if isinstance(s, dict) else s.lower() 
                                        for s in subcategories
                                    ]
                                    
                                    if new_subcat.lower() in subcategory_names:
                                        st.error(f"A subcategory named '{new_subcat}' already exists in this category.")
                                    else:
                                        # Ensure subcategories is a list of dicts
                                        if not isinstance(subcategories, list):
                                            subcategories = []
                                        
                                        # Convert to new format if needed
                                        if not all(isinstance(x, dict) for x in subcategories):
                                            subcategories = [
                                                {'name': subcat, 'color': '#D3D3D3'} 
                                                if isinstance(subcat, str) else subcat 
                                                for subcat in subcategories
                                            ]
                                        
                                        # Add new subcategory
                                        subcategories.append({
                                            'name': new_subcat,
                                            'color': subcat_color
                                        })
                                        
                                        # Update the categories dictionary
                                        categories_dict[selected_category]['subcategories'] = subcategories
                                        st.session_state['categories'] = categories_dict
                                        st.success(f"✅ Subcategory '{new_subcat}' added to '{selected_category}'!")
                                        st.rerun()
                                else:
                                    st.warning("Please enter a subcategory name")

            with tab_manage:
                st.markdown("### 🛠️ Manage Categories")
                
                if not categories_dict:
                    st.info("No categories available. Add some categories to get started!")
                else:
                    # Category selection
                    edit_category = st.selectbox(
                        "Select a category to edit or delete:",
                        ["Select a category..."] + sorted(list(categories_dict.keys())),
                        key='edit_category_select',
                        index=0
                    )
                    
                    if edit_category != "Select a category..." and edit_category in categories_dict:
                        st.markdown("---")
                        st.markdown(f"#### 🎨 Editing: **{edit_category}**")
                        
                        current_info = categories_dict[edit_category]
                        
                        # Category details in a card-like container
                        with st.container():
                            st.markdown("""
                                <style>
                                .category-card {
                                    padding: 1.5rem;
                                    border-radius: 0.5rem;
                                    background-color: #f8f9fa;
                                    border-left: 4px solid #4e79a7;
                                    margin-bottom: 1.5rem;
                                }
                                </style>
                                <div class='category-card'>
                                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                                        <h4 style='margin: 0;'>Category Details</h4>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Edit category name and color
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                new_name = st.text_input("Category Name", 
                                                       value=edit_category, 
                                                       key=f"edit_cat_name_{edit_category}")
                            with col2:
                                st.markdown("<div style='height: 30px; display: flex; align-items: flex-end;'>"
                                          "<div>Category Color</div></div>", 
                                          unsafe_allow_html=True)
                                new_color = st.color_picker("", 
                                                          value=current_info['color'],
                                                          key=f"edit_cat_color_{edit_category}",
                                                          label_visibility="collapsed")
                            
                            # Action buttons
                            col_save, col_del, _ = st.columns([1, 1, 3])
                            
                            with col_save:
                                if st.button("💾 Save Changes", 
                                           key=f"save_cat_{edit_category}",
                                           use_container_width=True):
                                    if new_name and new_name.strip() != edit_category:
                                        # If name changed, create new entry and delete old one
                                        categories_dict[new_name] = {
                                            'color': new_color,
                                            'subcategories': current_info['subcategories']
                                        }
                                        del categories_dict[edit_category]
                                        st.success(f"✅ Category renamed to '{new_name}'")
                                        st.session_state['edit_category_select'] = new_name
                                        st.session_state['categories'] = categories_dict
                                        st.rerun()
                                    else:
                                        # Just update color if name didn't change
                                        categories_dict[edit_category]['color'] = new_color
                                        st.success("✅ Category updated successfully!")
                                        st.session_state['categories'] = categories_dict
                                        st.rerun()
                            
                            with col_del:
                                if edit_category in default_categories:
                                    st.button("🔒 Default Category", 
                                             disabled=True,
                                             help="Default categories cannot be deleted",
                                             use_container_width=True)
                                else:
                                    if st.session_state.get('confirm_delete') != edit_category:
                                        if st.button("🗑️ Delete Category", 
                                                   key=f"delete_cat_{edit_category}",
                                                   type="secondary",
                                                   use_container_width=True):
                                            st.session_state['confirm_delete'] = edit_category
                                            st.rerun()
                                    else:
                                        if st.button("⚠️ Confirm Deletion", 
                                                   key=f"confirm_del_{edit_category}",
                                                   type="primary",
                                                   use_container_width=True):
                                            # Delete the category
                                            del categories_dict[edit_category]
                                            del st.session_state['confirm_delete']
                                            st.session_state['categories'] = categories_dict
                                            st.session_state['edit_category_select'] = "Select a category..."
                                            st.success(f"✅ Category '{edit_category}' deleted successfully!")
                                            st.rerun()
                            
                            if st.session_state.get('confirm_delete') == edit_category:
                                if st.button("❌ Cancel", 
                                           key=f"cancel_delete_{edit_category}",
                                           use_container_width=True):
                                    del st.session_state['confirm_delete']
                                    st.rerun()
                            
                            # Subcategories section
                            st.markdown("---")
                            st.markdown("#### 📋 Subcategories")
                            
                            if current_info['subcategories']:
                                st.markdown("""
                                    <style>
                                    .subcategory-item {
                                        padding: 0.5rem;
                                        margin: 0.25rem 0;
                                        border-radius: 0.25rem;
                                        background-color: #ffffff;
                                        border: 1px solid #e9ecef;
                                    }
                                    </style>
                                """, unsafe_allow_html=True)
                                
                                for i, subcat in enumerate(current_info['subcategories']):
                                    if isinstance(subcat, dict):
                                        sub_name = subcat['name']
                                        sub_color = subcat['color']
                                    else:
                                        sub_name = subcat
                                        sub_color = '#D3D3D3'
                                    
                                    st.markdown(f"<div class='subcategory-item'>", unsafe_allow_html=True)
                                    cols = st.columns([4, 2, 1, 1])
                                    with cols[0]:
                                        new_sub_name = st.text_input(
                                            "Name", 
                                            value=sub_name,
                                            key=f"sub_name_{edit_category}_{i}",
                                            label_visibility="collapsed"
                                        )
                                    with cols[1]:
                                        new_sub_color = st.color_picker(
                                            "Color",
                                            value=sub_color,
                                            key=f"sub_color_{edit_category}_{i}",
                                            label_visibility="collapsed"
                                        )
                                    with cols[2]:
                                        st.markdown("<div style='height: 30px; display: flex; align-items: center;'>"
                                                  f"<div style='width: 20px; height: 20px; background-color: {new_sub_color}; border: 1px solid #ddd; border-radius: 4px;'></div>"
                                                  "</div>", 
                                                  unsafe_allow_html=True)
                                    with cols[3]:
                                        if st.button("🗑️", 
                                                   key=f"del_sub_{edit_category}_{i}",
                                                   use_container_width=True):
                                            if len(current_info['subcategories']) > 1:
                                                current_info['subcategories'].pop(i)
                                                st.session_state['categories'] = categories_dict
                                                st.success("✅ Subcategory deleted successfully!")
                                                st.rerun()
                                            else:
                                                st.warning("A category must have at least one subcategory")
                                    
                                    # Update subcategory if changed
                                    if new_sub_name != sub_name or new_sub_color != sub_color:
                                        if isinstance(subcat, dict):
                                            subcat['name'] = new_sub_name
                                            subcat['color'] = new_sub_color
                                        else:
                                            current_info['subcategories'][i] = {
                                                'name': new_sub_name,
                                                'color': new_sub_color
                                            }
                                        st.session_state['categories'] = categories_dict
                                        st.rerun()
                                    
                                    st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Add new subcategory button
                            if st.button("➕ Add New Subcategory", 
                                       key=f"add_subcat_to_{edit_category}",
                                       use_container_width=True):
                                new_subcat = f"New Subcategory {len(current_info['subcategories']) + 1}"
                                new_color = get_distinct_color()
                                
                                # Ensure subcategories is a list of dicts
                                if not isinstance(current_info['subcategories'], list):
                                    current_info['subcategories'] = []
                                
                                current_info['subcategories'].append({
                                    'name': new_subcat,
                                    'color': new_color
                                })
                                st.session_state['categories'] = categories_dict
                                st.rerun()
            
            # Display current categories in a card view at the bottom
            st.markdown("---")
            st.markdown("### 📋 Current Categories")
            
            # Display categories in a grid
            cols = st.columns(2)
            for i, (cat, info) in enumerate(categories_dict.items()):
                with cols[i % 2]:
                    with st.container():
                        # Get the category color or use a default
                        cat_color = info.get('color', '#4e79a7')
                        
                        # Create a card for each category
                        st.markdown(f"""
                            <div style='
                                padding: 1rem;
                                margin-bottom: 1rem;
                                border-radius: 0.5rem;
                                background-color: #f8f9fa;
                                border-left: 4px solid {cat_color};
                            '>
                                <div style='
                                    display: flex;
                                    justify-content: space-between;
                                    align-items: center;
                                    margin-bottom: 0.5rem;
                                '>
                                    <h4 style='margin: 0;'>{cat}</h4>
                                    <div style='
                                        width: 24px;
                                        height: 24px;
                                        border-radius: 4px;
                                        background-color: {cat_color};
                                        border: 1px solid #ddd;
                                    '></div>
                                </div>
                                <div style='margin-top: 0.5rem;'>
                        """, unsafe_allow_html=True)
                        
                        # Display subcategories
                        subcats = info.get('subcategories', [])
                        if subcats:
                            for subcat in subcats:
                                if isinstance(subcat, dict):
                                    sub_name = subcat.get('name', 'Unnamed')
                                    sub_color = subcat.get('color', '#d3d3d3')
                                else:
                                    sub_name = subcat
                                    sub_color = '#d3d3d3'
                                    
                                st.markdown(f"""
                                    <div style='
                                        display: flex;
                                        justify-content: space-between;
                                        align-items: center;
                                        padding: 0.5rem;
                                        margin: 0.25rem 0;
                                        background-color: white;
                                        border-radius: 0.25rem;
                                        border: 1px solid #e9ecef;
                                        font-size: 0.9em;
                                    '>
                                        <span>{sub_name}</span>
                                        <div style='
                                            width: 16px;
                                            height: 16px;
                                            border-radius: 3px;
                                            background-color: {sub_color};
                                            border: 1px solid #ddd;
                                        '></div>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='padding: 0.5rem; color: #6c757d;'>No subcategories</div>", 
                                      unsafe_allow_html=True)
                        
                        st.markdown("""
                            </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
            
            # Add some space at the bottom
            st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

        with subtab_insights:
            # Show view-only mode message for viewers
            if st.session_state['user_role'] == 'viewer':
                st.info("🔍 You are in view-only mode. Viewers can see financial data but cannot make changes.")
            # Only restrict access if not admin/staff/viewer
            elif not can_edit_financials:
                st.info("🔒 You don't have permission to view financial insights. Please contact an administrator.")
            else:
                st.markdown("# 💡 Financial Insights")
            
            # Get data
            hog_data = st.session_state['hog_data'].dropna(subset=['Hog ID', 'Date', 'Weight (kg)'])
            
            # Expenses data
            expenses_df = st.session_state['financial_data'][
                st.session_state['financial_data']['Type'] == 'Expense'
            ].copy() if not st.session_state['financial_data'].empty else pd.DataFrame()
            
            # Calculate basic metrics
            num_hogs = hog_data['Hog ID'].nunique() if not hog_data.empty else 0
            total_expenses = expenses_df['Amount'].sum() if not expenses_df.empty else 0
            cost_per_hog = total_expenses / num_hogs if num_hogs > 0 else 0
            
            # Calculate weight gains
            if not hog_data.empty:
                hog_gains = hog_data.groupby('Hog ID').agg(
                    start_weight=('Weight (kg)', 'first'), 
                    end_weight=('Weight (kg)', 'last')
                )
                hog_gains['gain'] = hog_gains['end_weight'] - hog_gains['start_weight']
                total_gain = hog_gains['gain'].sum()
            else:
                total_gain = 0
                
            expense_per_kg_gain = total_expenses / total_gain if total_gain > 0 else 0
            
            # Get sales data if available
            sales_df = st.session_state['financial_data'][
                (st.session_state['financial_data']['Type'] == 'Sale') & 
                ~st.session_state['financial_data'].empty
            ].copy() if 'financial_data' in st.session_state and not st.session_state['financial_data'].empty else pd.DataFrame()
            total_revenue = sales_df['Amount'].sum() if not sales_df.empty else 0
            roi = ((total_revenue - total_expenses) / total_expenses * 100) if total_expenses > 0 else None

            # Key Metrics
            st.markdown("### 📊 Key Metrics")

            # Row 1: Expenses, Revenue, ROI
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            with row1_col1:
                st.metric("Total Expenses", format_number(total_expenses, 2, True))
            with row1_col2:
                st.metric("Total Revenue", format_number(total_revenue, 2, True))
            with row1_col3:
                if total_expenses > 0:
                    st.metric("ROI", f"{roi:.1f}%")
                else:
                    st.info("Add expenses to calculate ROI.")

            # Row 2: Number of Hogs, Cost per Hog, Expense per kg Gain, Total Weight Gain
            row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
            with row2_col1:
                st.metric("Number of Hogs", num_hogs)
            with row2_col2:
                st.metric("Cost per Hog", format_number(cost_per_hog, 2, True))
            with row2_col3:
                st.metric("Expense per kg Gain", format_number(expense_per_kg_gain, 2, True))
            with row2_col4:
                st.metric("Total Weight Gain", f"{total_gain:,.2f} kg")

            st.markdown("---")
            # Expense Category Breakdown
            if not expenses_df.empty:
                st.markdown("### 🥧 Expense Category Breakdown")
                category_summary = expenses_df.groupby('Category')['Amount'].agg(['sum', 'count']).reset_index()
                category_summary.columns = ['Category', 'Total Amount', 'Number of Expenses']
                category_summary = category_summary.sort_values('Total Amount', ascending=False)
                col1, col2 = st.columns(2)
                with col1:
                    for _, row in category_summary.iterrows():
                        percentage = (row['Total Amount'] / total_expenses) * 100 if total_expenses > 0 else 0
                        st.metric(
                            label=f"{row['Category']} ({row['Number of Expenses']} expenses)",
                            value=format_number(row['Total Amount'], 2, True),
                            delta=f"{percentage:.1f}% of total"
                        )
                with col2:
                    fig_pie = px.pie(
                        category_summary,
                        values='Total Amount',
                        names='Category',
                        title="Expense Distribution by Category",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Monthly Expense Trends
                st.markdown("### 📅 Monthly Expense Trends")
                expenses_df['Date'] = pd.to_datetime(expenses_df['Date'])
                expenses_df['Month'] = expenses_df['Date'].dt.to_period('M')
                monthly_expenses = expenses_df.groupby('Month')['Amount'].sum().reset_index()
                monthly_expenses['Month'] = monthly_expenses['Month'].astype(str)
                fig_line = px.line(
                    monthly_expenses,
                    x='Month',
                    y='Amount',
                    title="Monthly Expense Trends",
                    markers=True,
                    labels={'Amount': 'Total Amount (Kshs)', 'Month': 'Month'}
                )
                fig_line.update_traces(hovertemplate='Month: %{x}<br>Total: Kshs %{y:,.2f}')
                st.plotly_chart(fig_line, use_container_width=True)

            st.markdown("---")
            st.info("- Cost per hog: Total expenses divided by number of hogs.\n- Expense per kg gain: Total expenses divided by total weight gain.\n- ROI: (Revenue - Expenses) / Expenses × 100% (requires sales data).\n- For best accuracy, keep hog and expense records up to date.")

            # Sales Trends
            if not sales_df.empty:
                st.markdown("### 💰 Sales Trends")
                sales_df['Date'] = pd.to_datetime(sales_df['Date'])
                sales_df['Month'] = sales_df['Date'].dt.to_period('M')
                monthly_sales = sales_df.groupby('Month')['Amount'].sum().reset_index()
                monthly_sales['Month'] = monthly_sales['Month'].astype(str)
                fig_sales = px.line(
                    monthly_sales,
                    x='Month',
                    y='Amount',
                    title="Monthly Sales Trends",
                    markers=True,
                    labels={'Amount': 'Total Sales (Kshs)', 'Month': 'Month'}
                )
                fig_sales.update_traces(hovertemplate='Month: %{x}<br>Total: Kshs %{y:,.2f}')
                st.plotly_chart(fig_sales, use_container_width=True)

            st.markdown("---")
            st.info("- Cost per hog: Total expenses divided by number of hogs.\n- Expense per kg gain: Total expenses divided by total weight gain.\n- ROI: (Revenue - Expenses) / Expenses × 100% (requires sales data).\n- For best accuracy, keep hog and expense records up to date.")

        with subtab_sales:
            if not can_edit_financials:
                st.info("You do not have permission to add or edit sales records. Viewing only.")
                sales_df = st.session_state['financial_data'][st.session_state['financial_data']['Type'] == 'Sale'].copy() if not st.session_state['financial_data'].empty else pd.DataFrame()
                if not sales_df.empty:
                    st.subheader("All Sales Records")
                    sales_df_display = sales_df.copy()
                    sales_df_display['Date'] = pd.to_datetime(sales_df_display['Date']).dt.strftime('%d/%m/%Y')
                    st.dataframe(sales_df_display[['Date', 'Hog ID', 'Weight (kg)', 'Price/kg', 'Amount', 'Buyer', 'Description']], use_container_width=True, hide_index=True)
                else:
                    st.info("No sales records yet. Add sales above.")
            else:
                # --- Sales Tracking Section ---
                with st.container():
                    st.markdown("""
                    <div style='background-color:#f8f9fa;padding:15px;border-radius:10px;margin-bottom:20px;'>
                        <h2 style='color:#2c3e50;margin-bottom:15px;'>💰 Record New Sale</h2>
                        <p style='color:#6c757d;'>Enter the details of the hog sale below. All fields are required unless marked as optional.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.form(key='add_sale_form'):
                    # Create two columns for better layout
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### 📋 Sale Details")
                        sale_date = st.date_input(
                            "📅 Sale Date",
                            value=dt.date.today(),
                            key='sale_date',
                            help="Select the date when the sale was completed"
                        )
                        
                        # Hog IDs with better multi-select display
                        st.markdown("<div style='margin-top:10px;margin-bottom:5px;font-weight:500;'>🐷 Hog ID(s) Sold</div>", unsafe_allow_html=True)
                        hog_ids = st.multiselect(
                            "Select hogs sold in this transaction",
                            options=[f'{int(hid):03d}' for hid in st.session_state['hog_data']['Hog ID'].unique() if pd.notna(hid)],
                            key='sale_hog_ids',
                            help="Select all hogs included in this sale"
                        )
                        
                        # Weight and Price in a 2-column layout
                        weight_col, price_col = st.columns(2)
                        with weight_col:
                            sale_weight = st.number_input(
                                "⚖️ Weight (kg)",
                                min_value=0.0,
                                step=0.1,
                                format="%.1f",
                                key='sale_weight',
                                help="Total weight of all hogs sold"
                            )
                        with price_col:
                            price_per_kg = st.number_input(
                                "💰 Price/kg (Kshs)",
                                min_value=0.0,
                                step=10.0,
                                format="%.2f",
                                key='sale_price_per_kg',
                                help="Price per kilogram"
                            )
                        
                        # Auto-calculate total amount
                        calculated_amount = sale_weight * price_per_kg if sale_weight and price_per_kg else 0.0
                        sale_amount = st.number_input(
                            "💵 Total Sale Amount (Kshs)",
                            min_value=0.0,
                            step=100.0,
                            format="%.2f",
                            value=calculated_amount,
                            key='sale_amount',
                            help="This will auto-calculate but can be adjusted if needed"
                        )
                    
                    with col2:
                        st.markdown("### ℹ️ Additional Information")
                        
                        # Buyer information
                        buyer = st.text_input(
                            "👤 Buyer (Optional)",
                            key='sale_buyer',
                            placeholder="Enter buyer's name or company",
                            help="Helpful for record keeping"
                        )
                        
                        # Notes with larger text area
                        sale_notes = st.text_area(
                            "📝 Notes (Optional)",
                            key='sale_notes',
                            placeholder="Any additional details about this sale...",
                            height=100,
                            help="E.g., special conditions, payment terms, etc."
                        )
                        
                        # Submit button with better styling
                        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
                        submit_col1, submit_col2 = st.columns([1, 3])
                        with submit_col2:
                            submit_sale = st.form_submit_button(
                                "💾 Save Sale Record",
                                type="primary",
                                use_container_width=True
                            )
                    
                    # Form submission handling
                    if submit_sale:
                        if sale_date and hog_ids and sale_weight > 0 and price_per_kg > 0 and sale_amount > 0:
                            with st.spinner('Saving sale record...'):
                                new_sale = pd.DataFrame([{
                                    'Date': sale_date,
                                    'Type': 'Sale',
                                    'Category': 'Sale',
                                    'Description': f"Sold hog(s): {', '.join(hog_ids)}. {sale_notes}",
                                    'Hog ID': ','.join(hog_ids),
                                    'Weight (kg)': sale_weight,
                                    'Price/kg': price_per_kg,
                                    'Amount': sale_amount,
                                    'Buyer': buyer
                                }])
                                
                                # Ensure all required columns exist
                                for col in ['Date', 'Type', 'Category', 'Description', 'Hog ID', 'Weight (kg)', 'Price/kg', 'Amount', 'Buyer']:
                                    if col not in st.session_state['financial_data'].columns:
                                        st.session_state['financial_data'][col] = pd.NA
                                
                                # Add new sale and save
                                st.session_state['financial_data'] = pd.concat([st.session_state['financial_data'], new_sale], ignore_index=True)
                                save_financial_data(st.session_state['financial_data'])
                                
                                # Show success message
                                st.toast('✅ Sale record added successfully!', icon='✅')
                                st.rerun()
                        else:
                            st.error("❌ Please fill in all required fields and ensure values are greater than zero.")
            sales_df = st.session_state['financial_data'][st.session_state['financial_data']['Type'] == 'Sale'].copy() if not st.session_state['financial_data'].empty else pd.DataFrame()
            
            if not sales_df.empty:
                st.subheader("All Sales Records")
                
                # Format and display sales records
                sales_display = sales_df[['Date', 'Hog ID', 'Weight (kg)', 'Price/kg', 'Amount', 'Buyer', 'Description']].copy()
                sales_display['Date'] = pd.to_datetime(sales_display['Date']).dt.strftime('%d/%m/%Y')
                
                # Display with proper formatting
                st.dataframe(
                    sales_display,
                    column_config={
                        "Date": "Date",
                        "Hog ID": "Hog ID",
                        "Weight (kg)": st.column_config.NumberColumn(
                            "Weight (kg)",
                            format="%.1f"
                        ),
                        "Price/kg": st.column_config.NumberColumn(
                            "Price/kg (Kshs)",
                            format="%.2f"
                        ),
                        "Amount": st.column_config.NumberColumn(
                            "Total Amount (Kshs)",
                            format="%.2f"
                        ),
                        "Buyer": "Buyer",
                        "Description": "Description"
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add edit/delete section
                st.markdown("### ✏️ Edit or Delete Sale Record")
                
                # Create a list of sale records for selection
                sales_df['sale_label'] = sales_df.apply(
                    lambda row: f"{row['Date'].strftime('%Y-%m-%d')} | Hog(s): {row['Hog ID']} | {row['Weight (kg)']}kg @ Kshs {row['Price/kg']:,.2f}/kg | Total: Kshs {row['Amount']:,.2f}", 
                    axis=1
                )
                
                # Initialize session state for sale edit if not exists
                if 'sale_edit_select' not in st.session_state:
                    st.session_state.sale_edit_select = "Select a sale..."
                
                # Create a mapping of display labels to indices
                sale_options = ["Select a sale..."] + sales_df['sale_label'].tolist()
                
                # Display the selectbox with the current session state
                selected_sale = st.selectbox(
                    "Select a sale record to edit or delete:",
                    sale_options,
                    index=sale_options.index(st.session_state.sale_edit_select) if st.session_state.sale_edit_select in sale_options else 0,
                    key="sale_edit_select_widget"
                )
                
                # Update the session state with the current selection
                st.session_state.sale_edit_select = selected_sale
                
                if selected_sale and selected_sale != "Select a sale...":
                    # Find the selected sale
                    sale_idx = sales_df[sales_df['sale_label'] == selected_sale].index[0]
                    sale_data = sales_df.loc[sale_idx]
                    
                    # Action selection
                    action = st.radio(
                        "Action:",
                        ["Edit", "Delete"],
                        key=f"sale_action_{sale_idx}",
                        horizontal=True
                    )
                    
                    if action == "Edit":
                        with st.form(key=f'edit_sale_form_{sale_idx}'):
                            st.write("### ✏️ Edit Sale Record")
                            
                            # Pre-fill form with existing data
                            edit_date = st.date_input(
                                "Sale Date:",
                                value=pd.to_datetime(sale_data['Date']).date(),
                                key=f'edit_sale_date_{sale_idx}'
                            )
                            
                            # Parse hog IDs from the description
                            hog_id_text = sale_data['Description'].split(':')[1].split('.')[0].strip()
                            hog_ids = [id.strip() for id in hog_id_text.split(',')]
                            
                            edit_hog_ids = st.multiselect(
                                "Hog ID(s) Sold:",
                                [f'{int(hid):03d}' for hid in st.session_state['hog_data']['Hog ID'].unique() if pd.notna(hid)],
                                default=hog_ids,
                                key=f'edit_sale_hog_ids_{sale_idx}'
                            )
                            
                            edit_weight = st.number_input(
                                "Total Weight Sold (kg):",
                                min_value=0.0,
                                value=float(sale_data['Weight (kg)']),
                                format="%.2f",
                                key=f'edit_sale_weight_{sale_idx}'
                            )
                            
                            edit_price_per_kg = st.number_input(
                                "Price per kg (Kshs):",
                                min_value=0.0,
                                value=float(sale_data['Price/kg']),
                                format="%.2f",
                                key=f'edit_sale_price_per_kg_{sale_idx}'
                            )
                            
                            edit_amount = st.number_input(
                                "Total Sale Amount (Kshs):",
                                min_value=0.0,
                                value=float(sale_data['Amount']),
                                format="%.2f",
                                key=f'edit_sale_amount_{sale_idx}'
                            )
                            
                            edit_buyer = st.text_input(
                                "Buyer (optional):",
                                value=sale_data.get('Buyer', ''),
                                key=f'edit_sale_buyer_{sale_idx}'
                            )
                            
                            # Extract notes from description
                            notes = sale_data['Description'].split('.', 1)[1].strip() if '.' in sale_data['Description'] else ''
                            edit_notes = st.text_input(
                                "Notes (optional):",
                                value=notes,
                                key=f'edit_sale_notes_{sale_idx}'
                            )
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if st.form_submit_button("💾 Save Changes"):
                                    if edit_date and edit_hog_ids and edit_weight > 0 and edit_price_per_kg > 0 and edit_amount > 0:
                                        # Update the sale record
                                        st.session_state['financial_data'].at[sale_idx, 'Date'] = edit_date
                                        st.session_state['financial_data'].at[sale_idx, 'Hog ID'] = ','.join(edit_hog_ids)
                                        st.session_state['financial_data'].at[sale_idx, 'Weight (kg)'] = edit_weight
                                        st.session_state['financial_data'].at[sale_idx, 'Price/kg'] = edit_price_per_kg
                                        st.session_state['financial_data'].at[sale_idx, 'Amount'] = edit_amount
                                        st.session_state['financial_data'].at[sale_idx, 'Buyer'] = edit_buyer
                                        st.session_state['financial_data'].at[sale_idx, 'Description'] = f"Sold hog(s): {', '.join(edit_hog_ids)}. {edit_notes}"
                                        
                                        save_financial_data(st.session_state['financial_data'])
                                        st.success("Sale record updated successfully!")
                                        st.balloons()
                                        st.rerun()
                                    else:
                                        st.error("Please fill in all required fields with valid values.")
                            
                            with col2:
                                if st.form_submit_button("❌ Cancel"):
                                    if 'sale_edit_select' in st.session_state:
                                        del st.session_state.sale_edit_select
                                    st.rerun()
                    
                    elif action == "Delete":
                        st.warning("⚠️ Are you sure you want to delete this sale record? This action cannot be undone.")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("✅ Confirm Delete", key=f'confirm_delete_sale_{sale_idx}'):
                                # Remove the sale record
                                st.session_state['financial_data'] = st.session_state['financial_data'].drop(index=sale_idx).reset_index(drop=True)
                                
                                # Save the changes
                                save_financial_data(st.session_state['financial_data'])
                                st.success("Sale record deleted successfully!")
                                st.balloons()
                                if 'sale_edit_select' in st.session_state:
                                    del st.session_state.sale_edit_select
                                st.rerun()
                        
                        with col2:
                            if st.button("❌ Cancel", key=f'cancel_delete_sale_{sale_idx}'):
                                if 'sale_edit_select' in st.session_state:
                                    del st.session_state.sale_edit_select
                                st.rerun()
            else:
                st.info("No sales records yet. Add sales above.")

if __name__ == "__main__":
    main()