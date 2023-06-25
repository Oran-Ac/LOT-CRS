from model.BaseModel import BertCRS,BartCRS
import os
BACKBONE_MODEL_MAPPINGS = {
    "bert":"bert-base-uncased",
    "bart":"facebook/bart-base"
}
MODEL_TYPES_MAPPING = {
    "bert":BertCRS,
    "bart":BartCRS
}
Movie_Name_Path = {
    "redial":'/workspace/src/utils/movies_with_mentions.csv'
}
AT_TOKEN_NUMBER = {
    "redial":6924
}