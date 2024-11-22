import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def clean_rank_column(series):
    """
    Clean rank column by removing non-numeric characters
    and converting to numeric
    """
    return pd.to_numeric(series.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

def load_data_and_models():
    try:
        # Load dataset
        data = pd.read_csv("2022.csv")

        # Clean rank columns
        data["Opening Rank"] = clean_rank_column(data["Opening Rank"])
        data["Closing Rank"] = clean_rank_column(data["Closing Rank"])

        # Drop rows with NaN ranks
        data.dropna(subset=["Opening Rank", "Closing Rank"], inplace=True)

        # Encode categorical features
        label_encoders = {
            "Seat Type": LabelEncoder(),
            "Institute": LabelEncoder(),
            "Academic Program Name": LabelEncoder(),
            "Round": LabelEncoder(),
        }
        for col, encoder in label_encoders.items():
            data[col] = encoder.fit_transform(data[col].astype(str))

        # Extract features and targets
        X = data[["Seat Type", "Opening Rank", "Closing Rank"]]
        y_institute = data["Institute"]
        y_program = data["Academic Program Name"]
        y_round = data["Round"]

        # Initialize scaler and KNN model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        knn_institute = KNeighborsClassifier(n_neighbors=5)
        knn_program = KNeighborsClassifier(n_neighbors=5)
        knn_round = KNeighborsClassifier(n_neighbors=5)

        # Train models
        knn_institute.fit(X_scaled, y_institute)
        knn_program.fit(X_scaled, y_program)
        knn_round.fit(X_scaled, y_round)

        return (
            label_encoders,
            scaler,
            knn_institute,
            knn_program,
            knn_round,
            data,
        )
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.title("COLLEGE PREDICTION SYSTEM")
    st.write("ENTER YOUR SEAT TYPE AND RANK")

    # Load data and models
    result = load_data_and_models()
    if result is None:
        st.error("Could not load data. Please check the CSV file.")
        return

    label_encoders, scaler, knn_institute, knn_program, knn_round, data = result

    # User inputs
    seat_type = st.selectbox("Seat Type", label_encoders["Seat Type"].classes_)
    rank = st.number_input("Rank", min_value=0, step=1)

    # Predict button
    if st.button("Predict"):
        try:
            # Encode inputs
            encoded_seat_type = label_encoders["Seat Type"].transform([seat_type])[0]

            # Prepare input for prediction
            input_data = np.array([[encoded_seat_type, rank, rank]])
            scaled_input = scaler.transform(input_data)

            # Predictions
            institute_pred = knn_institute.predict(scaled_input)
            program_pred = knn_program.predict(scaled_input)
            round_pred = knn_round.predict(scaled_input)

            # Decode predictions
            institute_name = label_encoders["Institute"].inverse_transform(institute_pred)[0]
            program_name = label_encoders["Academic Program Name"].inverse_transform(program_pred)[0]
            round_name = label_encoders["Round"].inverse_transform(round_pred)[0]

            # Display predictions
            st.subheader("Predictions:")
            st.write(f"Predicted Institute: {institute_name}")
            st.write(f"Predicted Academic Program: {program_name}")
            st.write(f"Predicted Round: {round_name}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
