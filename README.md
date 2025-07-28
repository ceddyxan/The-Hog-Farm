# Hog Weight Tracking App

This is a standalone intelligent application for managing and analyzing hog weight records, built using Python and Streamlit.

## Features

- Add or remove hogs from the tracking list.
- Record weekly weight measurements with specific dates.
- Automatically display weights per hog over time.
- **Real-time, Interactive Analysis:**
    - Line chart showing weight trends per hog over time.
    - Average weight per week across all hogs.
    - Week-to-week growth tracking per hog (percentage and absolute increase).
    - Outlier detection: Highlight hogs that are significantly underperforming or overperforming.
    - Filter/search option by hog number or date.
- Export data to CSV/Excel.
- Automatically highlight hogs with no weight gain for two consecutive weeks.
- Provide an intelligent summary report with key insights.

## Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <repository_url>
    cd "Hog Farm App"
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the application, make sure your virtual environment is activated and then execute the following command in your terminal:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

## Data Storage

All hog weight data is stored locally in `hog_data.csv` in the same directory as `app.py`. You can directly inspect or edit this CSV file if needed.

## Tech Stack

-   **UI:** Streamlit
-   **Data Manipulation:** Pandas
-   **Charting:** Matplotlib
-   **Local Storage:** CSV 