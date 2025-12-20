# set up the base image
# (Keeping 3.12 as per your file, but ensure your CI/CD matches this version)
FROM python:3.12

# set the working directory
WORKDIR /app/

# copy the requirements file to workdir
COPY requirements.txt .

# install the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy all required data files at once
COPY ./data/collab_filtered_data.csv \
     ./data/interaction_matrix.npz \
     ./data/track_ids.npy \
     ./data/cleaned_data.csv \
     ./data/transformed_data.npz \
     ./data/transformed_hybrid_data.npz \
     ./data/

# Copy all required Python scripts at once
COPY app.py \
     collaborative_filtering.py \
     content_based_filtering.py \
     hybrid_recommendations.py \
     data_cleaning.py \
     transform_filtered_data.py \
     ./

# --- Disable Security Warnings for IP Access ---
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
# ---------------------------------------------------

# expose the port on the container
EXPOSE 8000

# run the streamlit app
CMD [ "streamlit", "run", "app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]