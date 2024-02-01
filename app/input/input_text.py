from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')


# Class which describes the input text that needs to be classified
class InputTextList(BaseModel):
    input_data_list: list  # The text to be classified
