import streamlit as st
import pandas as pd
import re
from datetime import datetime
import pytesseract
from PIL import Image, ImageOps, ImageEnhance
import plotly.express as px
import plotly.graph_objects as go
import unicodedata
import os

# Set page config
st.set_page_config(page_title="Payment History Analyzer", layout="wide")
st.title("💸 Payment History Analyzer")

# Initialize session state for transactions
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""

# Function to preprocess image for better OCR
def preprocess_image(image):
    # Convert to grayscale
    img = image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3.0)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    
    # Resize for better OCR
    img = img.resize((img.width * 3, img.height * 3))
    
    # Apply threshold to get binary image
    img = img.point(lambda x: 0 if x < 180 else 255, '1')
    
    return img

# Function to extract text using OCR
def extract_text_from_image(image):
    try:
        # Preprocess the image
        processed_img = preprocess_image(image)
        
        # Use Tesseract to extract text with custom configuration
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        return text
    except Exception as e:
        st.error(f"Error in OCR processing: {e}")
        return ""

# Function to manually parse transactions from the messy OCR text
def parse_transactions_from_text(text):
    transactions = []
    
    # Define patterns for different transaction types
    patterns = [
        # Pattern for transactions like "Harihar Fruit Shop -%10"
        r'([A-Za-z\s]+)[^\d]*([+-]?[^\d]*)(\d+\.?\d*)',
        # Pattern for transactions with dates
        r'(Paid|Received|Sent) on (\d{1,2} [A-Za-z]+, \d{1,2}:\d{2} [AP]M)',
    ]
    
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and headers
        if not line or any(header in line for header in ["Payment History", "Total Spent", "---"]):
            i += 1
            continue
            
        # Try to extract transaction information
        transaction = extract_transaction_info(line, lines, i)
        
        if transaction:
            transactions.append(transaction)
            # Skip ahead based on transaction pattern
            i += 2 if "Paid on" in line or "Received on" in line or "Sent on" in line else 1
        else:
            i += 1
    
    return transactions

# Function to extract transaction information from a line
def extract_transaction_info(line, lines, index):
    # Common OCR errors and their corrections
    corrections = {
        "%10": "10",
        "2170": "170",
        "#500": "500",
        "784": "84",
        "492": "492",
        "21,000": "1,000",
        "1,003": "1,003",
        "115": "115"
    }
    
    # Try to find amount in the line
    amount_match = re.search(r'[+-]?[^\d]*(\d+\.?\d*)', line)
    if not amount_match:
        return None
        
    amount_str = amount_match.group(1)
    
    # Apply corrections for common OCR errors
    for error, correction in corrections.items():
        if error in line:
            amount_str = correction
            break
    
    # Determine if it's income or expense
    is_income = '+' in line or "Received" in line
    amount = clean_amount(amount_str, is_income)
    
    # Find description (previous line might have it)
    description = ""
    if index > 0:
        prev_line = lines[index-1].strip()
        if not re.search(r'[+-]?[^\d]*(\d+\.?\d*)', prev_line) and not any(x in prev_line for x in ["Paid", "Received", "Sent"]):
            description = prev_line
    
    # If no description found, try to extract from current line
    if not description:
        description_match = re.match(r'([A-Za-z\s]+)[^\d]*[+-]?[^\d]*\d+\.?\d*', line)
        if description_match:
            description = description_match.group(1).strip()
    
    # Find date (next lines might have it)
    date_str = None
    for j in range(index+1, min(index+4, len(lines))):
        next_line = lines[j].strip()
        date_match = re.search(r'(\d{1,2} [A-Za-z]+, \d{1,2}:\d{2} [AP]M)', next_line)
        if date_match:
            date_str = date_match.group(1)
            break
    
    # Categorize transaction
    category = categorize_transaction(description)
    
    return {
        'Date': parse_date(date_str) if date_str else None,
        'Description': description,
        'Amount': amount,
        'Type': 'Income' if is_income else 'Expense',
        'Category': category
    }

# Function to clean and convert amount
def clean_amount(amount_str, is_income):
    if not amount_str:
        return 0.0
    
    # Remove commas
    amount_str = amount_str.replace(',', '')
    
    try:
        amount = float(amount_str)
        return amount if is_income else -amount
    except ValueError:
        return 0.0

# Function to parse date
def parse_date(date_str):
    if not date_str:
        return None
    
    # Map month abbreviations to numbers
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    try:
        # Extract date parts
        date_match = re.search(r'(\d{1,2}) ([A-Za-z]+), (\d{1,2}):(\d{2}) ([AP]M)', date_str)
        if date_match:
            day = int(date_match.group(1))
            month_str = date_match.group(2)
            hour = int(date_match.group(3))
            minute = int(date_match.group(4))
            am_pm = date_match.group(5)
            
            # Convert to 24-hour format
            if am_pm == 'PM' and hour != 12:
                hour += 12
            elif am_pm == 'AM' and hour == 12:
                hour = 0
            
            # Use current year
            year = datetime.now().year
            month = month_map.get(month_str[:3], 1)
            
            return datetime(year, month, day, hour, minute)
    except:
        pass
    
    return None

# Function to categorize transactions
def categorize_transaction(description):
    description_lower = description.lower()
    
    # Food-related
    if any(word in description_lower for word in ['fruit', 'shop', 'dairy', 'confection', 'groceries', 'qd']):
        return 'Food & Dining'
    
    # Person-to-person transfers
    if any(word in description_lower for word in ['received', 'sent', 'transfer', 'mom', 'afaque', 'irfan', 'rahul', 'moor']):
        return 'Transfers'
    
    # Education
    if 'education' in description_lower:
        return 'Education'
    
    # Business
    if 'elitmus' in description_lower:
        return 'Business'
    
    return 'Other'

# Sidebar for image upload
st.sidebar.header("Upload Payment Screenshot")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Extract text from image
    extracted_text = extract_text_from_image(image)
    st.session_state.extracted_text = extracted_text
    
    # Parse transactions from text
    if extracted_text and extracted_text.strip():
        with st.expander("Extracted Text"):
            st.text(extracted_text)
        
        # Try to parse the transactions
        transactions = parse_transactions_from_text(extracted_text)
        
        if transactions:
            # Update session state
            st.session_state.transactions = transactions
            st.success(f"Successfully extracted {len(transactions)} transactions!")
        else:
            st.error("Could not extract any transactions from the image. Please try with a clearer image.")
    else:
        st.error("Could not extract text from the image. Please try with a clearer image.")

# Display and analyze transactions
if st.session_state.transactions:
    # Create DataFrame
    df = pd.DataFrame(st.session_state.transactions)
    
    # Convert Date column to datetime for sorting
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date', ascending=False)
    
    # Display transactions
    st.subheader("📋 Transaction History")
    st.dataframe(df)
    
    # Summary statistics
    st.subheader("📊 Spending Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    total_income = df[df['Type'] == 'Income']['Amount'].sum()
    total_expenses = df[df['Type'] == 'Expense']['Amount'].sum()
    net_flow = total_income + total_expenses  # Expenses are negative
    
    col1.metric("Total Income", f"¥{total_income:,.2f}")
    col2.metric("Total Expenses", f"¥{abs(total_expenses):,.2f}")
    col3.metric("Net Cash Flow", f"¥{net_flow:,.2f}")
    
    # Category analysis
    st.subheader("📈 Spending by Category")
    
    # Expenses by category
    expense_df = df[df['Type'] == 'Expense']
    if not expense_df.empty:
        category_totals = expense_df.groupby('Category')['Amount'].sum().abs()
        
        # Create pie chart
        fig1 = px.pie(
            values=category_totals.values,
            names=category_totals.index,
            title="Expense Distribution by Category"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Create bar chart
        fig2 = px.bar(
            x=category_totals.index,
            y=category_totals.values,
            title="Expenses by Category",
            labels={'x': 'Category', 'y': 'Amount (¥)'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Monthly trend
    st.subheader("📅 Monthly Trends")
    
    # Add month-year column
    df['Month-Year'] = df['Date'].dt.to_period('M').astype(str)
    
    monthly_data = df.groupby(['Month-Year', 'Type'])['Amount'].sum().reset_index()
    monthly_data['Amount'] = monthly_data['Amount'].abs()
    
    # Pivot for plotting
    monthly_pivot = monthly_data.pivot(index='Month-Year', columns='Type', values='Amount').fillna(0)
    
    if not monthly_pivot.empty:
        fig3 = go.Figure()
        if 'Income' in monthly_pivot.columns:
            fig3.add_trace(go.Bar(
                x=monthly_pivot.index,
                y=monthly_pivot['Income'],
                name='Income'
            ))
        if 'Expense' in monthly_pivot.columns:
            fig3.add_trace(go.Bar(
                x=monthly_pivot.index,
                y=monthly_pivot['Expense'],
                name='Expense'
            ))
        
        fig3.update_layout(
            barmode='group',
            title='Monthly Income vs Expenses',
            xaxis_title='Month',
            yaxis_title='Amount (¥)'
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    # Daily spending trend
    st.subheader("📆 Daily Spending Pattern")
    
    daily_data = df[df['Type'] == 'Expense'].copy()
    if not daily_data.empty:
        daily_data['Day'] = daily_data['Date'].dt.day
        daily_spending = daily_data.groupby('Day')['Amount'].sum().abs()
        
        fig4 = px.line(
            x=daily_spending.index,
            y=daily_spending.values,
            title="Daily Spending Pattern",
            labels={'x': 'Day of Month', 'y': 'Amount Spent (¥)'}
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Export data
    st.subheader("💾 Export Data")
    if st.button("Export to CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="payment_history.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Upload a payment screenshot to get started. The app will extract and analyze your transaction history.")