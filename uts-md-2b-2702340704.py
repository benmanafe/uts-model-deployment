import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle as pkl

class ModelOOP:
    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.fitted_model = None
        self.num_cols = [
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
        self.cat_cols = [col for col in self.data.columns if col not in self.num_cols and col != 'booking_status' and col != 'Booking_ID']

    def data_split(self, target_column):
        self.X = self.data.drop([target_column], axis=1)
        self.y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
    
    def categorical_cleaning(self):
        self.X_train.loc[self.X_train['type_of_meal_plan'].isnull(), 'type_of_meal_plan'] = 'Meal Plan 1'
        self.X_test.loc[self.X_test['type_of_meal_plan'].isnull(), 'type_of_meal_plan'] = 'Meal Plan 1'
        
        self.X_train.loc[self.X_train['required_car_parking_space'].isnull(), 'required_car_parking_space'] = 0
        self.X_test.loc[self.X_test['required_car_parking_space'].isnull(), 'required_car_parking_space'] = 0
        
        self.X_train = self.X_train.drop(['Booking_ID'], axis=1)
        self.X_test = self.X_test.drop(['Booking_ID'], axis=1)
            
    def numerical_cleaning(self):
        avg_avg_price_per_room = round(self.X_train['avg_price_per_room'].mean(), 1)
        self.X_train.loc[self.X_train['avg_price_per_room'].isnull(), 'avg_price_per_room'] = avg_avg_price_per_room

    def scaleData(self):
        scaler = RobustScaler()

        preprocessor_numerical = ColumnTransformer(
            transformers=[
                ('scaler', scaler, self.num_cols),
            ],
            remainder='passthrough'
        )

        self.pipeline_numerical = Pipeline(steps=[('preprocessor', preprocessor_numerical)])

        X_train_scaled = self.pipeline_numerical.fit_transform(self.X_train[self.num_cols])
        X_test_scaled = self.pipeline_numerical.transform(self.X_test[self.num_cols])

        scaled_feature_names = self.num_cols
        all_numerical_feature_names = scaled_feature_names

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=all_numerical_feature_names, index=self.X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=all_numerical_feature_names, index=self.X_test.index)

        return X_train_scaled, X_test_scaled
    
    def encodeData(self):
        oe_cols = ['arrival_month',
                   'arrival_date']

        ohe_cols = ['type_of_meal_plan',
                    'required_car_parking_space',
                    'room_type_reserved',
                    'arrival_year',
                    'market_segment_type',
                    'repeated_guest']

        month_categories = list(range(1, 13))
        date_categories = list(range(1, 32))
        booking_status_categories = list(range(0, 2))

        ordinal_mapping = {
            'arrival_month': month_categories,
            'arrival_date': date_categories,
            'booking_status': booking_status_categories
        }

        oe = OrdinalEncoder(categories=[ordinal_mapping.get(col, sorted(self.X_train[col].unique())) for col in oe_cols])
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        preprocessor_categorical = ColumnTransformer(
            transformers=[
                ('ordinal', oe, oe_cols),
                ('onehot', ohe, ohe_cols)
            ],
            remainder='passthrough'
            )

        self.pipeline_categorical = Pipeline(steps=[('preprocessor', preprocessor_categorical)])

        X_train_encoded = self.pipeline_categorical.fit_transform(self.X_train[self.cat_cols])
        X_test_encoded = self.pipeline_categorical.transform(self.X_test[self.cat_cols])

        oe_feature_names = oe_cols
        ohe_feature_names = self.pipeline_categorical.named_steps['preprocessor'].named_transformers_['onehot'].get_feature_names_out(input_features=ohe_cols)

        X_train_encoded = pd.DataFrame(X_train_encoded, columns=oe_feature_names + ohe_feature_names.tolist(), index=self.X_train.index)
        X_test_encoded = pd.DataFrame(X_test_encoded, columns=oe_feature_names + ohe_feature_names.tolist(), index=self.X_test.index)

        return X_train_encoded, X_test_encoded                
        
    def data_preprocessing(self):
        X_train_scaled, X_test_scaled = self.scaleData()
        X_train_encoded, X_test_encoded = self.encodeData()
        
        X_train_scaled.reset_index(drop=True, inplace=True)
        X_test_scaled.reset_index(drop=True, inplace=True)
        X_train_encoded.reset_index(drop=True, inplace=True)
        X_test_encoded.reset_index(drop=True, inplace=True)
        self.X_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)
        self.y_test.reset_index(drop=True, inplace=True)

        
        self.X_train = pd.concat([X_train_encoded, X_train_scaled], axis=1)
        self.X_test = pd.concat([X_test_encoded, X_test_scaled], axis=1)
        
        self.y_train = self.y_train.map({'Canceled': 1, 'Not_Canceled': 0})
        self.y_test = self.y_test.map({'Canceled': 1, 'Not_Canceled': 0})
        
        self.target_vals = {'Canceled': 1, 'Not_Canceled': 0}
        
        return self.target_vals
    
    def train(self, model, param_grid):
        self.fitted_model = GridSearchCV(model, param_grid, cv=5)
        self.fitted_model.fit(self.X_train, self.y_train)
    
    def evaluate(self, pred):
        print(f'Classification Report :\n{classification_report(self.y_test, pred)}')
        
    def test(self):
        pred = self.fitted_model.predict(self.X_test)
        self.evaluate(pred)
    
    def best_params(self):
        if self.fitted_model == None:
            raise ValueError("No Model Found!")
        return self.fitted_model.best_params_
    
    def export_model(self, filename):
        if self.fitted_model == None:
            raise ValueError("No Model Found!")
        with open(filename, 'wb') as file:
            pkl.dump(self.fitted_model, file)

    def export_scaler(self, filename):
        with open(filename, 'wb') as file:
            pkl.dump(self.pipeline_numerical, file)
        
    def export_encoder(self, filename):
        with open(filename, 'wb') as file:
            pkl.dump(self.pipeline_categorical, file)
    
    def export_target_vals(self, filename):
        with open(filename, 'wb') as file:
            pkl.dump(self.target_vals, file)
            
def main():
    model = ModelOOP(data='Dataset_B_hotel.csv')
    model.data_split(target_column='booking_status')
    data_targets = model.data_preprocessing()
    print(f'Data Targets: {data_targets}')

    param_grid = {
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [300, 500, 700],
    }
    
    rf_model = XGBClassifier()
    model.train(model=rf_model, param_grid=param_grid)
    model.test()
    model.best_params()

    model.export_model('xgb_model.pkl')
    model.export_scaler('scaler.pkl')
    model.export_encoder('encoder.pkl')
    model.export_target_vals('target_vals.pkl')

if __name__ == '__main__':
    main()