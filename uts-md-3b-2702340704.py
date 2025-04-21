import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import datetime
import time

def main():
    st.title('Machine Learning Booking Status App')
    st.subheader('Name: Benjamin Eleazar Manafe')
    st.subheader('NIM: 2702340704')
    st.info('This app will predict a customer\'s booking status (Canceled/Not Canceled)!')
    
    with st.expander('**Data**'):
        data = pd.read_csv('Dataset_B_hotel.csv')
        data = data.dropna()
        st.write('This is a raw data')
        st.dataframe(data)
    
    current_date_data = datetime.date.today()
    current_date = st.date_input('When is the order date?', value=current_date_data)
    
    no_of_adults = st.number_input(
        'How many adults?', value=0, placeholder='Type amount of adults...'
    )
    
    no_of_children = st.number_input(
        'How many children?', value=0, placeholder='Type amount of childrens...'
    )
    
    if no_of_children == 0 and no_of_adults == 0:
        no_of_adults = 1
    
    no_of_days_staying = st.number_input(
        'How long will you be staying (in days)?', value=1, placeholder='Type amount of nights...'
    )
    
    arrival_date_data = datetime.date(current_date_data.year + 1, current_date_data.month, current_date_data.day)
    arrival_date = st.date_input('When is the arrival date?', value=arrival_date_data)
    
    lead_time = (arrival_date_data - current_date).days
    arrival_year = arrival_date.year
    arrival_month = arrival_date.month
    arrival_day = arrival_date.day
    
    no_of_weekend_nights = 0
    no_of_week_nights = 0

    for i in range(no_of_days_staying):
        arr_date = arrival_date + datetime.timedelta(days=i)
        day_of_week = arr_date.weekday()
        
        if day_of_week >= 5:
            no_of_weekend_nights += 1
        else:
            no_of_week_nights += 1
        
    type_of_meal_plan_data = sorted(data['type_of_meal_plan'].unique())
    type_of_meal_plan = st.selectbox(
        'What is your type of meal plan choosing?', 
        type_of_meal_plan_data,
    )
    
    required_car_parking_space_data = ['Yes', 'No']
    required_car_parking_space = st.selectbox(
        'Is a car parking space required?',
        required_car_parking_space_data
    )
    
    if required_car_parking_space == 'Yes':
        required_car_parking_space_new = 1
    else:
        required_car_parking_space_new = 0
    
    room_type_reserved_data = sorted(data['room_type_reserved'].unique())
    room_type_reserved  = st.selectbox(
        'Is a care parking space required?',
        room_type_reserved_data
    )
    
    market_segment_type_data = sorted(data['market_segment_type'].unique())
    market_segment_type = st.selectbox(
        'What is your market segment type?',
        market_segment_type_data
    )
    
    repeated_guest_data = ['Yes', 'No']
    repeated_guest = st.selectbox(
        'Are you a revisiting guest?',
        repeated_guest_data 
    )
    
    if repeated_guest == 'Yes':
        repeated_guest_new = 1
    else:
        repeated_guest_new = 0
    
    no_of_previous_cancellations = st.number_input(
        'How many times have this order been cancelled before this?', value=0, placeholder='Type amount of previous cancellations...'
    )
    
    no_of_previous_bookings_not_canceled = st.number_input(
        'How many times have this order has not been canceled before this?', value=0, placeholder='Type amount of previous bookings not canceled...'
    )
    
    average_price_by_room = data.groupby('room_type_reserved')['avg_price_per_room'].mean()
    avg_price_per_room = st.number_input(
        'Average price per room:', value=average_price_by_room[room_type_reserved], placeholder='Type average price per room...'
    )
    
    no_of_special_requests = st.number_input(
        'Number of special requests:', value=0, placeholder='Type number of special requests...'
    )
    
    if 'user_data' not in st.session_state:
        st.session_state['user_data'] = None
    
    if st.button('Load Data'):
        st.session_state['user_data'] = pd.DataFrame([{
            'no_of_adults': no_of_adults,
            'no_of_children': no_of_children,
            'no_of_weekend_nights': no_of_weekend_nights,
            'no_of_week_nights': no_of_week_nights,
            'type_of_meal_plan': type_of_meal_plan,
            'required_car_parking_space': required_car_parking_space,
            'room_type_reserved': room_type_reserved,
            'lead_time': lead_time,
            'arrival_year': arrival_year,
            'arrival_month': arrival_month,
            'arrival_date': arrival_day,
            'market_segment_type': market_segment_type,
            'repeated_guest': repeated_guest,
            'no_of_previous_cancellations': no_of_previous_cancellations,
            'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
            'avg_price_per_room': avg_price_per_room,
            'no_of_special_requests': no_of_special_requests
        }])
    
    if st.session_state['user_data'] is not None:
        st.write('Data')
        st.dataframe(st.session_state['user_data'].T)
        
        st.session_state['user_data'] = pd.DataFrame([{
            'no_of_adults': no_of_adults,
            'no_of_children': no_of_children,
            'no_of_weekend_nights': no_of_weekend_nights,
            'no_of_week_nights': no_of_week_nights,
            'type_of_meal_plan': type_of_meal_plan,
            'required_car_parking_space': required_car_parking_space_new,
            'room_type_reserved': room_type_reserved,
            'lead_time': lead_time,
            'arrival_year': arrival_year,
            'arrival_month': arrival_month,
            'arrival_date': arrival_day,
            'market_segment_type': market_segment_type,
            'repeated_guest': repeated_guest_new,
            'no_of_previous_cancellations': no_of_previous_cancellations,
            'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
            'avg_price_per_room': avg_price_per_room,
            'no_of_special_requests': no_of_special_requests
        }])
        
        if (st.button('Predict')):
            progress_text = 'Predicting...'
            bar = st.progress(0, text=progress_text)

            for percent in range(100):
                time.sleep(1e-4)
                bar.progress(percent + 1, text=progress_text)
            time.sleep(1)
            bar.empty()
                
            make_predictions(data=st.session_state['user_data'])
        else:
            st.info('Please load the data first by clicking the \'Load Data\' button.')


def scale_data(data, features):
    with open('scaler.pkl', 'rb') as file:
        loaded_scaler = pkl.load(file)
    
    new_numerical_scaled = loaded_scaler.transform(data[features])
    new_numerical_scaled_df = pd.DataFrame(new_numerical_scaled, columns=features)
    
    return new_numerical_scaled_df
    
def encode_data(data, features, oe, ohe):
    with open('encoder.pkl', 'rb') as file:
        loaded_encoder = pkl.load(file)
    
    new_categorical_encoded = loaded_encoder.transform(data[features])
    
    encoded_feature_names = []
    for name, transformer, columns in loaded_encoder.named_steps['preprocessor'].transformers_:
        if name == 'ordinal':
            encoded_feature_names.extend(columns)
        elif name == 'onehot':
            encoded_feature_names.extend(transformer.get_feature_names_out(input_features=columns))
    
    new_categorical_encoded_df = pd.DataFrame(new_categorical_encoded, columns=encoded_feature_names)
    
    return new_categorical_encoded_df, encoded_feature_names

def preprocess_data(data):
    num_cols = [
        'no_of_adults', 
        'no_of_children',
        'no_of_weekend_nights',
        'no_of_week_nights',
        'lead_time',
        'no_of_previous_cancellations',
        'no_of_previous_bookings_not_canceled',
        'avg_price_per_room',
        'no_of_special_requests'
        ]
    cat_cols = [col for col in data.columns if col not in num_cols and col != 'booking_status' and col != 'Booking_ID']
    
    oe_cols = ['arrival_month',
               'arrival_date']
    ohe_cols = [col for col in cat_cols if col not in oe_cols]
    
    scaled_data = scale_data(data=data.copy(), features=num_cols)
    encoded_data, encoded_feature_names = encode_data(data=data.copy(), features=cat_cols, oe=oe_cols, ohe=ohe_cols)
    
    processed_columns_order = encoded_feature_names + num_cols
    processed_data = pd.concat([encoded_data, scaled_data], axis=1)[processed_columns_order]
    return processed_data

def make_predictions(data):
    with open('xgb_model.pkl', 'rb') as file:
        loaded_model = pkl.load(file)

    with open('target_vals.pkl', 'rb') as file:
        loaded_target_vals = pkl.load(file)
    
    processed_data = preprocess_data(data=data)
    predictions = loaded_model.predict(processed_data)
    print(predictions)
    inverse_target_vals = {v: k for k, v in loaded_target_vals.items()}
    prediction_probs = loaded_model.predict_proba(processed_data)
    st.dataframe(pd.DataFrame(prediction_probs, columns=inverse_target_vals.values()))
    
    if predictions[0] == 0:
        st.success(f'The predicted output is: {predictions[0]} **[{inverse_target_vals[predictions[0]]}]**')
    else:
        st.error(f'The predicted output is: {predictions[0]} **[{inverse_target_vals[predictions[0]]}]**')
    
if __name__ == '__main__':
    main()
