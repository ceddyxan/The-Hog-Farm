import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt
import plotly.express as px

DATA_FILE = 'hog_data.csv'

st.set_page_config(layout="wide")

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
    st.title("Hog Weight Tracking App")

    if 'hog_data' not in st.session_state:
        st.session_state['hog_data'] = load_data()

    # --- Sidebar for Hog Management, Deletion, and Import ---
    with st.sidebar.expander("Hog Management"):
        hog_id = st.text_input("Enter Hog ID to add:", "", key='add_hog_id_input')
        
        # Placeholder for add hog confirmation pop-up
        add_hog_placeholder = st.empty()

        if add_hog_placeholder.button("Add Hog", key='add_hog_button'):
            if hog_id:
                # Check if hog already exists before asking for confirmation
                if hog_id not in st.session_state['hog_data']['Hog ID'].astype(str).unique():
                    st.session_state['confirm_add_hog'] = True
                    st.session_state['hog_id_to_add'] = hog_id
                    st.rerun()
                else:
                    add_hog_placeholder.warning(f"Hog {int(hog_id):03d} already exists.")
            else:
                add_hog_placeholder.warning("Please enter a Hog ID.")

        if st.session_state.get('confirm_add_hog', False):
            with add_hog_placeholder.container():
                st.warning(f"Are you sure you want to add Hog: {int(st.session_state['hog_id_to_add']):03d}?")
                col_confirm_add_hog_yes, col_confirm_add_hog_no = st.columns(2)
                with col_confirm_add_hog_yes:
                    if st.button("Yes, Add Hog", key='confirm_add_hog_yes'):
                        new_hog_df = pd.DataFrame([{'Hog ID': int(st.session_state['hog_id_to_add']), 'Date': None, 'Weight (kg)': None}])
                        st.session_state['hog_data'] = pd.concat([st.session_state['hog_data'], new_hog_df], ignore_index=True)
                        save_data(st.session_state['hog_data'])
                        st.success(f"Hog {int(st.session_state['hog_id_to_add']):03d} added!")
                        del st.session_state['confirm_add_hog']
                        del st.session_state['hog_id_to_add']
                        st.rerun()
                with col_confirm_add_hog_no:
                    if st.button("No, Cancel", key='confirm_add_hog_no'):
                        del st.session_state['confirm_add_hog']
                        del st.session_state['hog_id_to_add']
                        st.info("Hog addition cancelled.")
                        st.rerun()

        formatted_hog_ids = [f'{int(hid):03d}' for hid in st.session_state['hog_data']['Hog ID'].unique() if pd.notna(hid)]
        hogs_to_remove_display = st.multiselect("Select Hog(s) to remove:", formatted_hog_ids, key='remove_hog_multiselect')

        # Placeholder for remove hog confirmation pop-up
        remove_hog_placeholder = st.empty()

        if hogs_to_remove_display:
            if remove_hog_placeholder.button("Remove Selected Hog(s)", key='remove_hog_button'):
                st.session_state['confirm_remove_hog'] = True
                st.session_state['hogs_to_remove_display'] = hogs_to_remove_display # Store for rerun
                st.rerun()

        if st.session_state.get('confirm_remove_hog', False):
            with remove_hog_placeholder.container():
                st.warning(f"Are you sure you want to remove Hog(s): {', '.join(st.session_state['hogs_to_remove_display'])}? This action cannot be undone.")
                col_confirm_remove_yes, col_confirm_remove_no = st.columns(2)
                with col_confirm_remove_yes:
                    if st.button("Yes, Remove", key='confirm_remove_hog_yes'):
                        hogs_to_remove_int = [int(hid) for hid in st.session_state['hogs_to_remove_display']]
                        st.session_state['hog_data'] = st.session_state['hog_data'][~st.session_state['hog_data']['Hog ID'].isin(hogs_to_remove_int)]
                        st.session_state['hog_data'].dropna(subset=['Date', 'Weight (kg)'], inplace=True) 
                        save_data(st.session_state['hog_data'])
                        st.success(f"Hog(s) {', '.join(st.session_state['hogs_to_remove_display'])} removed.")
                        del st.session_state['confirm_remove_hog']
                        del st.session_state['hogs_to_remove_display']
                        st.rerun()
                with col_confirm_remove_no:
                    if st.button("No, Cancel", key='confirm_remove_hog_no'):
                        del st.session_state['confirm_remove_hog']
                        del st.session_state['hogs_to_remove_display']
                        st.info("Hog removal cancelled.")
                        st.rerun()

    with st.sidebar.expander("Delete Weight Records"):
        delete_date = st.date_input("Select Date to Delete Records From:", value=None, key='delete_date_input')
        delete_hog_ids = st.multiselect("Select Hog(s) to Delete Records For (Optional):",
                                            st.session_state['hog_data']['Hog ID'].unique(),
                                            key='delete_hog_ids_multiselect')
        
        # Placeholder for delete records confirmation pop-up
        delete_records_placeholder = st.empty()

        if delete_date or delete_hog_ids:
            if delete_records_placeholder.button("Delete Selected Weight Records", key='delete_records_button'):
                st.session_state['confirm_delete_records'] = True
                st.session_state['delete_date'] = delete_date # Store for rerun
                st.session_state['delete_hog_ids'] = delete_hog_ids # Store for rerun
                st.rerun()

        if st.session_state.get('confirm_delete_records', False):
            with delete_records_placeholder.container():
                display_delete_date = st.session_state['delete_date'] if st.session_state['delete_date'] else 'All Dates'
                display_delete_hog_ids = ', '.join([str(int(hid)) for hid in st.session_state['delete_hog_ids']]) if st.session_state['delete_hog_ids'] else 'All Hogs'
                st.warning(f"Are you sure you want to delete records for Date: {display_delete_date} and Hog(s): {display_delete_hog_ids}? This action cannot be undone.")
                col_confirm_delete_yes, col_confirm_delete_no = st.columns(2)
                with col_confirm_delete_yes:
                    if st.button("Yes, Delete", key='confirm_delete_records_yes'):
                        initial_row_count = len(st.session_state['hog_data'])
                        df = st.session_state['hog_data'].copy()

                        if st.session_state['delete_date']:
                            df = df[df['Date'] != st.session_state['delete_date']]
                        
                        if st.session_state['delete_hog_ids']:
                            df = df[~df['Hog ID'].isin(st.session_state['delete_hog_ids'])]

                        deleted_row_count = initial_row_count - len(df)
                        st.session_state['hog_data'] = df
                        save_data(st.session_state['hog_data'])

                        if deleted_row_count > 0:
                            st.success(f"{deleted_row_count} record(s) deleted.")
                        else:
                            st.info("No records found matching the criteria to delete.")
                        del st.session_state['confirm_delete_records']
                        del st.session_state['delete_date']
                        del st.session_state['delete_hog_ids']
                        st.rerun()
                with col_confirm_delete_no:
                    if st.button("No, Cancel", key='confirm_delete_records_no'):
                        del st.session_state['confirm_delete_records']
                        del st.session_state['delete_date']
                        del st.session_state['delete_hog_ids']
                        st.info("Deletion cancelled.")
                        st.rerun()

    with st.sidebar.expander("Import Data"):
        uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key='file_uploader')
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

    # Filter/Search options (placed in sidebar)
    st.sidebar.header("Filter Data") # Keep header here
    with st.sidebar.expander("Filter Data Options"):
        search_hog_id = st.text_input("Search by Hog ID:", "", key='sidebar_search_hog_id')
        search_start_date = st.date_input("Search from Date:", value=None, key='sidebar_start_date')
        search_end_date = st.date_input("Search to Date:", value=None, key='sidebar_end_date')

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Data Entry & View", "Analysis", "Summary Report"])

    with tab1:
        st.header("Record Weight Measurement")
        
        available_hogs = st.session_state['hog_data']['Hog ID'].unique()
        # Filter out None values and format Hog IDs for display
        available_hogs_formatted = [f'{int(hog):03d}' for hog in available_hogs if pd.notna(hog)]

        if len(available_hogs_formatted) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_hog_id_formatted = st.selectbox("Select Hog ID:", available_hogs_formatted, key='record_hog_id')
                # Convert selected formatted ID back to original int for processing
                selected_hog_id = int(selected_hog_id_formatted) if selected_hog_id_formatted else None
            with col2:
                measurement_date = st.date_input("Measurement Date:", key='record_date')
            with col3:
                weight = st.number_input("Weight (kg):", min_value=0.0, format="%.2f", key='record_weight')

            st.markdown("") # Add a bit of space

            # Placeholder for add weight record confirmation pop-up
            add_weight_placeholder = st.empty()

            if add_weight_placeholder.button("Add Weight Record", key='add_record_button'):
                if selected_hog_id and measurement_date and weight > 0:
                    st.session_state['confirm_add_weight'] = True
                    st.session_state['selected_hog_id'] = selected_hog_id
                    st.session_state['measurement_date'] = measurement_date
                    st.session_state['weight'] = weight
                    st.session_state['selected_hog_id_formatted'] = selected_hog_id_formatted
                    st.rerun()
                else:
                    add_weight_placeholder.warning("Please fill in all fields (Hog ID, Date, Weight) and ensure weight is greater than 0.")

            if st.session_state.get('confirm_add_weight', False):
                with add_weight_placeholder.container():
                    st.warning(f"Are you sure you want to add a weight record for Hog: {st.session_state['selected_hog_id_formatted']} on {st.session_state['measurement_date']} with Weight: {st.session_state['weight']:.2f} kg?")
                    col_confirm_weight_yes, col_confirm_weight_no = st.columns(2)
                    with col_confirm_weight_yes:
                        if st.button("Yes, Add Record", key='confirm_add_weight_yes'):
                            new_record = pd.DataFrame([{'Hog ID': st.session_state['selected_hog_id'], 'Date': st.session_state['measurement_date'], 'Weight (kg)': st.session_state['weight']}])
                            st.session_state['hog_data'] = pd.concat([st.session_state['hog_data'], new_record], ignore_index=True)
                            st.session_state['hog_data'].drop_duplicates(subset=['Hog ID', 'Date'], inplace=True, keep='last')
                            st.session_state['hog_data'].sort_values(by=['Hog ID', 'Date'], inplace=True)
                            save_data(st.session_state['hog_data'])
                            st.success(f"Weight record added for Hog {st.session_state['selected_hog_id_formatted']} on {st.session_state['measurement_date']}.")
                            del st.session_state['confirm_add_weight']
                            del st.session_state['selected_hog_id']
                            del st.session_state['measurement_date']
                            del st.session_state['weight']
                            del st.session_state['selected_hog_id_formatted']
                            st.rerun()
                    with col_confirm_weight_no:
                        if st.button("No, Cancel", key='confirm_add_weight_no'):
                            del st.session_state['confirm_add_weight']
                            del st.session_state['selected_hog_id']
                            del st.session_state['measurement_date']
                            del st.session_state['weight']
                            del st.session_state['selected_hog_id_formatted']
                            st.info("Weight record addition cancelled.")
                            st.rerun()
        else:
            st.info("No hogs added yet. Please add a hog using the sidebar.")

        st.markdown("--- ") # Separator

        st.header("Hog Weight Data")
        if not st.session_state['hog_data'].empty:
            # Filter out rows with None in 'Date' or 'Weight (kg)'
            display_data = st.session_state['hog_data'].dropna(subset=['Hog ID', 'Date', 'Weight (kg)'])
            
            if not display_data.empty:
                # Pivot the data for display as requested by the user
                # Convert 'Date' to datetime objects first for proper pivoting and formatting
                pivot_data = display_data.copy()
                pivot_data['Date'] = pd.to_datetime(pivot_data['Date'])
                pivot_data_display = pivot_data.pivot_table(index='Hog ID', columns='Date', values='Weight (kg)', aggfunc='first')
                
                # Sort columns (dates) in descending order
                pivot_data_display = pivot_data_display.iloc[:, ::-1]

                # Format column headers to resemble "D/M/YYYY" or "DD/MM/YYYY"
                pivot_data_display.columns = pivot_data_display.columns.strftime('%d/%m/%Y')
                
                # Rename the columns index to 'WEEK'
                pivot_data_display.columns.name = 'WEEK'

                # Format Hog ID index to be at least three digits
                pivot_data_display.index = pivot_data_display.index.map(lambda x: f'{int(x):03d}')
                pivot_data_display.index.name = 'Hog ID'

                st.dataframe(pivot_data_display, use_container_width=True)

                # Export data
                st.subheader("Export Data")
                col_export1, col_export2 = st.columns(2)
                with col_export1:
                    csv_file = display_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Export as CSV",
                        data=csv_file,
                        file_name='hog_weight_data.csv',
                        mime='text/csv',
                    )
                with col_export2:
                    # Streamlit's download_button requires bytes for Excel, so we need to save to a buffer first
                    import io
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        # Define sheet names
                        pivot_sheet_name = 'Pivoted Weight Data'
                        raw_sheet_name = 'Raw Hog Data'

                        # Write pivoted data to the first sheet
                        # Start at row 2 (0-indexed) to leave room for the title and a blank row
                        pivot_data_display.to_excel(writer, sheet_name=pivot_sheet_name, startrow=2)
                        
                        # Get the worksheet object to write the title
                        worksheet_pivot = writer.sheets[pivot_sheet_name]
                        title_format = writer.book.add_format({'bold': True, 'font_size': 14})
                        worksheet_pivot.write('A1', 'Hog Weight Data - Weekly Overview', title_format)

                        # Write raw data to the second sheet
                        display_data.to_excel(writer, index=False, sheet_name=raw_sheet_name)
                    excel_file_bytes = buffer.getvalue()
                    st.download_button(
                        label="Export as Excel",
                        data=excel_file_bytes,
                        file_name='hog_weight_data.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    )

            else:
                st.info("No complete weight records to display or plot yet.")
        else:
            st.info("No hog data available yet.")

    with tab2:
        st.header("Analysis")

        # Average weight per week across all hogs
        if not display_data.empty:
            display_data_copy = display_data.copy()
            display_data_copy['Date'] = pd.to_datetime(display_data_copy['Date'])
            display_data_copy['Week'] = display_data_copy['Date'].dt.isocalendar().week.astype(int)
            display_data_copy['Year'] = display_data_copy['Date'].dt.year
            
            avg_weight_per_week = display_data_copy.groupby(['Year', 'Week'])['Weight (kg)'].mean().reset_index()
            avg_weight_per_week['YearWeek'] = avg_weight_per_week['Year'].astype(str) + "-W" + avg_weight_per_week['Week'].astype(str).str.zfill(2)
            avg_weight_per_week = avg_weight_per_week.sort_values(by='YearWeek')
            
            if not avg_weight_per_week.empty:
                st.subheader("Average Weight Per Week (All Hogs)")
                # Use Plotly for interactive plot with hover
                fig_avg_plotly = px.line(avg_weight_per_week,
                                        x='YearWeek',
                                        y='Weight (kg)',
                                        title="Average Weight Per Week Across All Hogs",
                                        markers=True,
                                        hover_data={'YearWeek':False, 'Weight (kg)':':.1fkg'})
                
                fig_avg_plotly.update_traces(hovertemplate='Week: %{x}<br>Average Weight: %{y:.1f}kg')
                fig_avg_plotly.update_xaxes(tickangle=45)

                st.plotly_chart(fig_avg_plotly, use_container_width=True)

        st.markdown("--- ") # Separator
        # Week-to-week growth tracking per hog
        st.subheader("Week-to-Week Growth Per Hog")
        growth_data = []
        for hog_id in display_data['Hog ID'].unique():
            hog_data = display_data[display_data['Hog ID'] == hog_id].sort_values(by='Date')
            if len(hog_data) > 1:
                hog_data['Previous Weight (kg)'] = hog_data['Weight (kg)'].shift(1)
                hog_data['Weight Change (kg)'] = hog_data['Weight (kg)'] - hog_data['Previous Weight (kg)']
                hog_data['Growth (%)'] = (hog_data['Weight Change (kg)'] / hog_data['Previous Weight (kg)']) * 100
                
                # Highlight hogs with no weight gain for two consecutive weeks
                hog_data['No Gain Flag'] = (hog_data['Weight Change (kg)'] <= 0).rolling(window=2).sum() == 2

                growth_data.append(hog_data.dropna(subset=['Weight Change (kg)', 'Growth (%)']))
        
        if growth_data:
            all_growth_data = pd.concat(growth_data)
            
            # Summary of Growth Data
            st.subheader("Growth Summary")
            if not all_growth_data.empty:
                avg_growth_percent = all_growth_data['Growth (%)'].mean()
                num_hogs_positive_growth = all_growth_data[all_growth_data['Weight Change (kg)'] > 0]['Hog ID'].nunique()
                num_hogs_negative_growth = all_growth_data[all_growth_data['Weight Change (kg)'] < 0]['Hog ID'].nunique()

                col_avg, col_pos, col_neg = st.columns(3)

                with col_avg:
                    st.metric(label="Overall Average Growth", value=f"{avg_growth_percent:.1f}%")
                with col_pos:
                    st.info(f"Hogs with Positive Growth: {num_hogs_positive_growth}")
                with col_neg:
                    st.info(f"Hogs with Negative Growth: {num_hogs_negative_growth}")

                # Top 3 Best Performers
                with st.expander("View Top 3 Best Performers"):
                    if not all_growth_data.empty and not all_growth_data['Growth (%)'].empty:
                        top_performers = all_growth_data.nlargest(3, 'Growth (%)')
                        st.success("**Top 3 Best Performers:**")
                        for index, row in top_performers.iterrows():
                            st.success(f"- Hog #{int(row['Hog ID']):03d} gained {row['Growth (%)']:.1f}% on {row['Date'].strftime('%d/%m/%Y')}")
                    else:
                        st.info("Not enough data to determine top performers.")

                # Top 3 Least Performers (most negative weight change)
                with st.expander("View Top 3 Least Performers"):
                    negative_growth_hogs_data = all_growth_data[all_growth_data['Weight Change (kg)'] < 0]
                    if not negative_growth_hogs_data.empty:
                        least_performers = negative_growth_hogs_data.nsmallest(3, 'Weight Change (kg)')
                        st.error("**Top 3 Least Performers (with negative growth):**")
                        for index, row in least_performers.iterrows():
                            st.error(f"- Hog #{int(row['Hog ID']):03d} lost {abs(row['Weight Change (kg)']):.1f} kg ({row['Growth (%)']:.1f}% growth) on {row['Date'].strftime('%d/%m/%Y')}")
                    else:
                        st.info("No hogs showed negative growth.")

            else:
                st.info("No growth data available for summary.")

            st.subheader("Hogs with No Weight Gain (Two Consecutive Weeks)")
            no_gain_hogs = all_growth_data[all_growth_data['No Gain Flag'] == True].copy() # Make a copy to avoid SettingWithCopyWarning
            if not no_gain_hogs.empty:
                no_gain_hogs['Hog ID'] = no_gain_hogs['Hog ID'].apply(lambda x: f'{int(x):03d}')
                st.warning("The following hogs have shown no weight gain for two consecutive weeks:")
                st.dataframe(no_gain_hogs[['Hog ID', 'Date', 'Weight (kg)', 'Weight Change (kg)']], hide_index=True, use_container_width=True)
            else:
                st.info("No hogs detected with no weight gain for two consecutive weeks.")

            st.subheader("Hogs with Negative Growth")
            with st.expander("View Hogs with Negative Growth"):
                negative_growth_all_hogs = all_growth_data[all_growth_data['Weight Change (kg)'] < 0].copy()
                if not negative_growth_all_hogs.empty:
                    negative_growth_all_hogs['Hog ID'] = negative_growth_all_hogs['Hog ID'].apply(lambda x: f'{int(x):03d}')
                    st.dataframe(negative_growth_all_hogs[['Hog ID', 'Date', 'Growth (%)']], hide_index=True, use_container_width=True)
                else:
                    st.info("No hogs showed negative growth in any week.")

        st.markdown("--- ") # Separator
        st.subheader("Detailed Growth Data")
        with st.expander("View Detailed Growth Data"):
            display_growth_data = all_growth_data.copy()
            display_growth_data['Hog ID'] = display_growth_data['Hog ID'].apply(lambda x: f'{int(x):03d}')

            # Add a selection box for Hog ID
            unique_hog_ids_growth = sorted(display_growth_data['Hog ID'].unique())
            
            # Add a placeholder option at the beginning
            selection_options = ["Select a Hog ID"] + list(unique_hog_ids_growth)

            selected_hog_id_for_growth = st.selectbox(
                "Select Hog ID to view detailed growth data:",
                selection_options,
                index=0, # Set default to the placeholder
                key='detailed_growth_hog_select'
            )

            if selected_hog_id_for_growth != "Select a Hog ID":
                filtered_detailed_growth = display_growth_data[display_growth_data['Hog ID'] == selected_hog_id_for_growth]
                if not filtered_detailed_growth.empty:
                    st.dataframe(filtered_detailed_growth[['Hog ID', 'Date', 'Weight (kg)', 'Weight Change (kg)', 'Growth (%)']], hide_index=True, use_container_width=True)
                else:
                    st.info(f"No detailed growth data for Hog {selected_hog_id_for_growth}.")

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
            if not all_growth_data.empty and all_growth_data['Weight Change (kg)'].std() > 0:
                mean_growth = all_growth_data['Weight Change (kg)'].mean()
                std_growth = all_growth_data['Weight Change (kg)'].std()
                
                # Define an outlier as more than 2 standard deviations from the mean
                all_growth_data['Is Outlier'] = (
                    (all_growth_data['Weight Change (kg)'] > mean_growth + 2 * std_growth) |
                    (all_growth_data['Weight Change (kg)'] < mean_growth - 2 * std_growth)
                )
                
                outliers = all_growth_data[all_growth_data['Is Outlier']].copy() # Make a copy
                if not outliers.empty:
                    outliers['Hog ID'] = outliers['Hog ID'].apply(lambda x: f'{int(x):03d}')
                    st.warning("Hogs with significant weight change (Outliers):")
                    st.dataframe(outliers[['Hog ID', 'Date', 'Weight (kg)', 'Weight Change (kg)', 'Growth (%)']], hide_index=True, use_container_width=True)
                else:
                    st.info("No significant outliers detected in weight change this week.")
            else:
                st.info("Not enough data or variation to detect outliers effectively.")

        st.markdown("--- ") # Separator
        # The filter data section should only be once in the sidebar
        # These filtering operations should apply to the displayed data in the main area of tab2
        filtered_data = display_data.copy()
        if search_hog_id:
            # Filter by formatted Hog ID if user inputs formatted string
            filtered_data['Formatted Hog ID'] = filtered_data['Hog ID'].apply(lambda x: f'{int(x):03d}')
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
        # Let's pass the current session state data to a helper to derive them.

        # Recalculate display_data and all_growth_data if not available globally (e.g., if tabs executed independently)
        # This assumes display_data and all_growth_data are meant to reflect the latest state.
        current_hog_data = st.session_state['hog_data']
        display_data_summary = current_hog_data.dropna(subset=['Hog ID', 'Date', 'Weight (kg)'])

        growth_data_summary = []
        if not display_data_summary.empty:
            for hog_id in display_data_summary['Hog ID'].unique():
                hog_data_g = display_data_summary[display_data_summary['Hog ID'] == hog_id].sort_values(by='Date')
                if len(hog_data_g) > 1:
                    hog_data_g['Previous Weight (kg)'] = hog_data_g['Weight (kg)'].shift(1)
                    hog_data_g['Weight Change (kg)'] = hog_data_g['Weight (kg)'] - hog_data_g['Previous Weight (kg)']
                    hog_data_g['Growth (%)'] = (hog_data_g['Weight Change (kg)'] / hog_data_g['Previous Weight (kg)']) * 100
                    growth_data_summary.append(hog_data_g.dropna(subset=['Weight Change (kg)', 'Growth (%)']))

        all_growth_data_summary = pd.concat(growth_data_summary) if growth_data_summary else pd.DataFrame()

        summary_report = generate_summary_report(display_data_summary, all_growth_data_summary)
        
        # Display the improved summary report
        if 'status' in summary_report:
            st.info(summary_report['status'])
        else:
            st.subheader("Overall Summary")
            st.info(f"Total Hogs Tracked: {summary_report['overall']['total_hogs']}")
            
            if 'average_daily_gain' in summary_report['overall']:
                st.info(f"Overall Average Daily Gain (ADG): {summary_report['overall']['average_daily_gain']:.2f} kg/day")

            if 'best_overall_performer' in summary_report['overall']:
                best_overall = summary_report['overall']['best_overall_performer']
                st.success(f"Overall Best Performer (ADG): Hog #{best_overall['id']}: {best_overall['adg']:.2f} kg/day")

            if 'least_overall_performer' in summary_report['overall']:
                least_overall = summary_report['overall']['least_overall_performer']
                st.error(f"Overall Least Performer (ADG): Hog #{least_overall['id']}: {least_overall['adg']:.2f} kg/day")

            st.markdown("--- ") # Separator
            st.subheader(f"Latest Week Summary (as of {summary_report['latest_week_date']})")
            if 'growth_status' in summary_report['latest_week']:
                st.info(summary_report['latest_week']['growth_status'])
            else:
                if summary_report['latest_week']['average_herd_gain'] is not None:
                    st.info(f"Average herd gain this week: {summary_report['latest_week']['average_herd_gain']:.1f} kg.")
                
                if summary_report['latest_week']['highest_performer']:
                    hp = summary_report['latest_week']['highest_performer']
                    st.success(f"Highest Performer: Hog #{hp['id']}: {hp['weight_change']:.1f} kg ({hp['growth_percent']:.1f}% growth).")
                
                if summary_report['latest_week']['least_performer']:
                    lp = summary_report['latest_week']['least_performer']
                    st.error(f"Least Performer: Hog #{lp['id']}: {lp['weight_change']:.1f} kg ({lp['growth_percent']:.1f}% growth).")

                if summary_report['latest_week']['no_growth_hogs']:
                    no_growth_hog_ids = ", ".join([f'{int(hid):03d}' for hid in summary_report['latest_week']['no_growth_hogs']])
                    st.warning(f"Hog #{no_growth_hog_ids} showed no growth this week.")

            # Observations deduced from data
            st.subheader("Observations")
            observations_made = False

            if summary_report['potential_health_concern_hogs']:
                for concern in summary_report['potential_health_concern_hogs']:
                    st.warning(f"- Hog #{concern['id']}: {concern['consecutive_measurements']} consecutive non-positive weight changes as of {concern['latest_date']}.")
                observations_made = True

            if summary_report['latest_week'].get('no_growth_hogs'):
                no_growth_hog_ids_obs = ", ".join([f'{int(hid):03d}' for hid in summary_report['latest_week']['no_growth_hogs']])
                st.info(f"- Hog #{no_growth_hog_ids_obs} showed no growth in the latest week.")
                observations_made = True
            
            if summary_report['latest_week'].get('least_performer'):
                lp_obs = summary_report['latest_week']['least_performer']
                st.info(f"- Hog #{lp_obs['id']} was the least performer this week with {lp_obs['weight_change']:.1f} kg change.")
                observations_made = True

            if not observations_made:
                st.info("No critical observations detected this period.")


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


if __name__ == "__main__":
    main() 