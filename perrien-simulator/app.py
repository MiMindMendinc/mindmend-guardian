import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Perrien Simulator",
    page_icon="🔬",
    layout="wide"
)

# Title and description
st.title("Perrien Simulator")
st.markdown("""
The Perrien Simulator is a powerful tool designed to simulate various scenarios for analysis and visualization.
It provides users with an intuitive interface to interact with the simulation models, enabling them to explore
different outcomes based on the parameters input.
""")

# Sidebar for controls
st.sidebar.header("Controls")

# GPU toggle
use_gpu = st.sidebar.checkbox("Use GPU", value=False)

if use_gpu:
    st.sidebar.success("GPU mode enabled")
else:
    st.sidebar.info("CPU mode active")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualization")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV
            df = pd.read_csv(uploaded_file)
            
            # Display data info
            st.write(f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Create a sample visualization
            if len(df.columns) >= 2:
                fig = px.scatter(
                    df,
                    x=df.columns[0],
                    y=df.columns[1],
                    title="Data Visualization"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dataset needs at least 2 columns for visualization")
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        # Show sample data when no file is uploaded
        sample_data = pd.DataFrame({
            'X': np.random.randn(100),
            'Y': np.random.randn(100)
        })
        fig = px.scatter(
            sample_data,
            x='X',
            y='Y',
            title="Sample Data (Upload CSV to see your data)"
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Data Preview")
    
    if uploaded_file is not None:
        # Show data table
        st.dataframe(df.head(10), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Processed CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
    else:
        st.info("No data uploaded yet. Use the file uploader in the sidebar.")

# Footer
st.markdown("---")
st.markdown("Perrien Simulator | MindMend Guardian Project")
