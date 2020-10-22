class MLServerError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class InvalidMLLibFormat(MLServerError):
    def __init__(self, name: str, model_uri: str = None):
        msg = f"Invalid MLLib format for model {name}"
        if model_uri:
            msg += f" ({model_uri})"

        super().__init__(msg)
