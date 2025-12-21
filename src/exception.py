import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Returns detailed error message with file name and line number
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = (
        f"Error occurred in file [{file_name}] "
        f"at line [{line_number}] "
        f"with error message: {str(error)}"
    )

    return error_message


class CustomException(Exception):
    """
    Custom exception class for the project
    """

    def __init__(self, error: Exception, error_detail: sys):
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message
