from sklearn.preprocessing import MinMaxScaler


class ScalerFactory:
    @staticmethod
    def create_scaler(model_type: str):
        if model_type == "lstm":
            return MinMaxScaler()
        else:
            return None
