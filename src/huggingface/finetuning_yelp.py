import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import logging, DefaultDataCollator
from datasets import load_dataset


logging.set_verbosity(logging.ERROR)
dataset = load_dataset("yelp_review_full")
print(dataset)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_datasets = dataset.map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=True
)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

data_collator = DefaultDataCollator(return_tensors="tf")
tf_train_dataset = small_train_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
tf_test_dataset = small_test_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)

model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)

# model.fit(tf_train_dataset, epochs=1, batch_size=128, validation_data=tf_test_dataset, verbose=1)
