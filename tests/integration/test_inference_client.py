from fraud_detection_contracts.inference import (
    CardFeatures,
    FraudDetectionInferenceInput,
    Transaction,
)

from fraud_detection_processor.inference_client import predict

def test_predict():
    payload = FraudDetectionInferenceInput(
        transaction=Transaction(
            amt=258.37,
            lat=40.6761,
            long=-91.0391,
            city_pop=725,
            merch_lat=39.748629,
            merch_long=-90.81943,
            merchant="fraud_Skiles LLC",
            category="home",
            gender="F",
            state="IL",
            job="Environmental consultant",
        ),
        card_features=CardFeatures(
            distance_km=20.0,
            card_transaction_count=10,
            card_fraud_count=1,
            card_fraud_rate=0.1,
            card_avg_amount=100.0,
        ),
    )

    result = predict(payload)
    print(result)
    
    assert 0.0 <= result.fraud_probability <= 1.0
    assert isinstance(result.fraud_prediction, bool)