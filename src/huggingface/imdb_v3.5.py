import numpy as np
import pandas as pd
import tensorflow as tf
import os
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFAutoModel
from transformers import logging, DataCollatorWithPadding
from transformers import create_optimizer
from datasets import load_dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.set_verbosity(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts, labels = [], []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir / label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)
    return texts, labels


# train_texts, train_labels = read_imdb_split("../data/aclImdb/train")
# test_texts, test_labels = read_imdb_split("../data/aclImdb/test")

# train_df = pd.DataFrame({"texts": train_texts, "labels": train_labels})
# test_df = pd.DataFrame({"texts": test_texts, "labels": test_labels})

# train_df.to_pickle("../data/aclImdb/train.pkl")
# test_df.to_pickle("../data/aclImdb/test.pkl")

train_df = pd.read_pickle("../data/aclImdb/train.pkl")
test_df = pd.read_pickle("../data/aclImdb/test.pkl")
print(len(train_df), len(test_df))
print(max([len(sent) for sent in train_df["texts"]]))
print(np.mean([len(sent) for sent in train_df["texts"]]))
print(train_df["labels"].value_counts())

train_df = train_df[:1000]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = TFAutoModel.from_pretrained("distilbert-base-uncased")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
# print(inputs)
# outputs = model(inputs)
# print(outputs)

# last_hidden_states = outputs.last_hidden_state


epochs = 1
batch_size = 128
max_len = 40


def make_review_data(df):
    input_ids, attention_masks, token_type_ids, labels = [], [], [], []
    for sentence in tqdm(df["texts"]):
        encoded_sentence = tokenizer.encode_plus(
            text=sentence,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        input_ids.append(encoded_sentence["input_ids"])
        attention_masks.append((encoded_sentence["attention_mask"]))
        token_type_ids.append(encoded_sentence["token_type_ids"])

    input_ids = np.array(input_ids, dtype=np.int32)
    attention_masks = np.array(attention_masks, dtype=np.int32)
    token_type_ids = np.array(attention_masks, dtype=np.int32)
    labels = np.array(df["labels"], dtype=np.int32)
    return (input_ids, attention_masks, token_type_ids), labels


x_train, y_train = make_review_data(train_df)
# x_test, y_test = make_review_data(test_df)
print("input_ids: ", x_train[0])
print("attention_mask", x_train[1])


def format_dataset(input_ids, attention_masks, labels):
    return {"input_ids": input_ids, "attention_mask": attention_masks}, labels


datasets = tf.data.Dataset.from_tensor_slices((x_train[0], x_train[1], y_train))
datasets = datasets.map(format_dataset, num_parallel_calls=4)
for data in datasets.take(1):
    print(data)


# train_texts, val_texts, train_labels, val_labels = train_test_split(
#     train_df["texts"].values, train_df["labels"].values, test_size=0.2
# )
# test_texts, test_labels = zip(*test_df.values.tolist())
# print(test_texts[0])
# print(test_labels[0])


# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
# val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)
# test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)
# print(train_encodings[0])

# train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
# print(train_dataset[0])

# val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))
# test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels))

# model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
# loss = tf.keras.losses.SparceCategoricalCrossentropy(from_logits=True)
# model.compile(optimizer=optimizer, loss=loss)
# model.fit(train_dataset.shuffle(1000).batch(16), epochs=1, batch_size=16)


# class TFBertClassifier(tf.keras.Model):
#     def __init__(self, model_name, cache_dir, num_class):
#         super().__init__()
#         self.bert = TFBertModel.from_pretrained(model_name, cache_dir=cache_dir)
#         self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
#         self.classifier = tf.keras.layers.Dense(
#             num_class,
#             kernel_initializer=tf.keras.initializers.TruncatedNormal(
#                 self.bert.config.initializer_range
#             ),
#         )

#     def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
#         # outputs: sequence_output, pooled_output, (hidden_states), (attentions)
#         outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         pooled_output = outputs[1]
#         pooled_output = self.dropout(pooled_output, training=training)
#         logits = self.classifier(pooled_output)
#         return logits


# model = TFBertClassifier(model_name=model_name, cache_dir=cache_dir, num_class=2)

# optimizer = tf.keras.optimizers.Adam(3e-5)
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
# model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


# callbacks = [
#     EarlyStopping(monitor="val_accuracy", min_delta=0.0001, patience=2),
#     ModelCheckpoint(
#         "../data/bert_nsmc_weights.h5",
#         monitor="val_accuracy",
#         verbose=1,
#         save_best_only=True,
#         save_weights_only=True,
#     ),
# ]

# history = model.fit(
#     x_train,
#     y_train,
#     epochs=epochs,
#     batch_size=batch_size,
#     validation_split=0.1,
#     callbacks=callbacks,
# )

# plt.figure(figsize=(10, 6))
# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"], "")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend(["loss", "val_loss"])
# plt.savefig("images/bert_nsmc", dpi=300)


# results = model.evaluate(x_test, y_test, batch_size=1024)
# print("test loss, test acc: ", results)
