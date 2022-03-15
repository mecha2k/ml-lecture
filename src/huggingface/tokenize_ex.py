import tensorflow as tf

from transformers import AutoModel, AutoTokenizer
from transformers import logging

logging.set_verbosity(logging.ERROR)


# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# encoded_input = tokenizer(
#     "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
# )
# print(encoded_input)
# print(tokenizer.decode(encoded_input["input_ids"]))

# batch_sentences = [
#     "But what about second breakfast?",
#     "Don't think he knows about second breakfast, Pip.",
#     "What about elevensies?",
# ]
# encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
# print(encoded_input)


model = AutoModel.from_pretrained("klue/bert-base")
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

batch_sentences = [
    "But what about second breakfast?",
    "15일 한국부동산원에 따르면 올해 2월 주택종합(아파트·연립주택·단독주택) 매매가격 동향을 조사한 결과",
    "서울은 전월 대비 0.04%, 수도권은 0.03% 하락했다.",
]
encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="tf")
print(encoded_input["input_ids"])
print(encoded_input["input_ids"][0])
print(tokenizer.decode(encoded_input["input_ids"][1]))
print(tokenizer.decode(encoded_input["input_ids"][2]))
