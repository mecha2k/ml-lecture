import numpy as np
import tensorflow as tf
import os
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import logging, DataCollatorWithPadding
from transformers import create_optimizer
from datasets import load_dataset


logging.set_verbosity(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

imdb = load_dataset("imdb")
print(imdb)
print(imdb["train"][0])


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_imdb = imdb.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
print(tokenized_imdb)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
tf_train_dataset = tokenized_imdb["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)
tf_validation_dataset = tokenized_imdb["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)


batch_size = 16
num_epochs = 5
batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(
    init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps
)


model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model.compile(optimizer=optimizer)

# model.fit(x=tf_train_dataset, validation_data=tf_validation_dataset, epochs=1, verbose=1)
