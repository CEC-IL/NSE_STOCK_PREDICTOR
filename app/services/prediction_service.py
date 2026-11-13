import os
from sqlalchemy.orm import Session
from app.models.stock import DailyPrice
from app.schemas.stock import PredictionResponse, FeatureDriver

class PredictionService:
    #Generates predictions using either the ML model or the baseline model

    def __init__(self, model_path: str = None):
        self.model = None

        #TODO: Load the ML model from the specified path when provided
        if model_path and os.path.exists(model_path):
            import joblib
            self.model = joblib.load(model_path)
            print(f"ML model loaded from: {model_path}")

    def predict_for_stock(self, symbol: str, db: Session ) -> PredictionResponse:
        #Produces a prediction for the given stock symbol

        #Fetch the most recent 20 training days
        history = (
            db.query(DailyPrice)
            .filter(DailyPrice.symbol == symbol )
            .order_by(DailyPrice.date.desc())
            .limit(20)
            .all()
        )

        #handle case wher stock has no data
        if not history:
            return PredictionResponse(
                symbol=symbol,
                prediction_direction = "UNCHANGED",
                up_probability = 50.0,
                expected_return_range = "-0.0% to +0.0%",
                confidence_level = "Low",
                top_drivers = [
                    FeatureDriver(
                        feature_name = "No data",
                        impact = "neutral",
                        description = "Insufficient training data for this symbol"
                    )
                ]
            )

        #get the most recent training day data
        latest = history[0]
        drivers = []
        probability = 50.0
        is_bullish = False

        #short term trend on the moving averages
        if latest.sma_5d is not None and latest.sma_20 is not None:
            if latest.sma_5d > latest.sma_20:
                is_bullish = True
                probability += 10.0
                drivers.append(
                    FeatureDriver(
                        feature_name = "Short-term Trend (SMA 5 vs SMA 20)",
                        impact = "bullish",
                        description = (
                            f"5-day average (KES {latest.sma_5:.2f}) is above "
                            f"20-day average (KES {latest.sma_20:.2f}), "
                            f"suggesting upward momentum."
                        )
                    )
                )
            else:
                probability -= 8.0
                drivers.append(
                    FeatureDriver(
                        feature_name = "Short-term Trend (SMA 5 vs SMA 20)",
                        impact = "bearish",
                        description = (
                            f"5-day average (KES {latest.sma_5:.2f}) is below "
                            f"20-day average (KES {latest.sma_20:.2f}), "
                            f"suggesting downward pressure."
                        )
                    )
                )

        #Recent daily return momentum
        if latest.daily_return is not None:
            if latest.daily_return > 1.0:
                probability += 5.0
                drivers.append(
                    FeatureDriver(
                        feature_name = "Recent Price Momentum",
                        impact = "bullish",
                        description = (
                            f"Last trading day showed a {latest.daily_return:.2f}% gain, "
                            f"indicating buying interest."
                        )
                    )
                )
            elif latest.daily_return < -1.0:
                probability -= 5.0
                drivers.append(
                    FeatureDriver(
                        feature_name = "Recent Price Momentum",
                        impact = "bearish",
                        description = (
                            f"Last trading day showed a {latest.daily_return:.2f}% loss, "
                            f"indicating selling pressure."
                        )
                    )
                )
            else:
                drivers.append(
                    FeatureDriver(
                        feature_name = "Recent Price Momentum",
                        impact = "neutral",
                        description = (
                            f"Last trading day moved {latest.daily_return:.2f}%, "
                            f"within normal range."
                        )
                    )
                )

        #volatility risk assesment
        if latest.volatility_20d is not None:
            if latest.volatility_20d > 2.0:
                probability -= 4.0
                drivers.append(
                    FeatureDriver(
                        feature_name = "20-Day Volatility",
                        impact = "bearish",
                        description = (
                            f"Elevated price volatility ({latest.volatility_20d:.2f}%) "
                            f"widens uncertainty and increases risk."
                        )
                    )
                )
            else:
                drivers.append(
                    FeatureDriver(
                        feature_name = '20-Day Volatility',
                        impact = "neutral",
                        description = (
                            f"Moderate price volatility ({latest.volatility_20d:.2f}%), "
                            f"suggesting stable trading conditions."
                        )
                    )
                )

        #clamp probability between 10% and 90%
        probability = max(10.0, min(90.0, probability))

        #Determine direction and  confidence
        if probability >= 55.0:
            direction = "UP"
        elif probability <= 45.0:
            direction = "DOWN"
        else:
            direction = "UNCHANGED"

        if abs(probability - 50.0) >= 15:
            confidence = "High"
        elif abs(probability - 50.0) >= 5:
            confidence = "Moderate"
        else:
            confidence = "Low"

        #calculate expected return range based on volatility
        vol = latest.volatility_20d if latest.volatility_20d else 1.5
        lower_bound =  -1.2 * vol
        upper_bound = 1.4 * vol if is_bullish else 0.8 * vol

        return PredictionResponse(
            symbol=symbol,
            target_window = "5 Trading Days",
            prediction_direction = direction,
            up_probability = round( probability, 1),
            expected_return_range = f"{lower_bound:+.1f}% to {upper_bound:+.1f}%",
            confidence_level = confidence,
            top_drivers = drivers 
        )

#create a single shared instance of the service
prediction_service = PredictionService()