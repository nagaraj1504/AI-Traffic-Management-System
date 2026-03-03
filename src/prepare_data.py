import pandas as pd

def prepare_dataset():
    df = pd.read_csv("data/Train.csv")

    # Convert datetime
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Feature engineering
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Encode categorical
    df['weather_type'] = df['weather_type'].astype('category').cat.codes
    df['weather_description'] = df['weather_description'].astype('category').cat.codes
    df['is_holiday'] = df['is_holiday'].astype('category').cat.codes

    # Drop datetime
    df = df.drop(columns=['date_time'])

    df = df.dropna()

    df.to_csv("data/final_dataset.csv", index=False)
    print("✅ Final dataset created")

if __name__ == "__main__":
    prepare_dataset()