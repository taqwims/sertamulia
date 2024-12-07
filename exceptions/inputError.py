from exceptions.client_error import ClientError

class InputError(ClientError):
    def __init__(self, message):
        super().__init__(message)
        self.name = "InputError"
