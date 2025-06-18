# Train mô hình CTC Transformer

import os
import tensorflow as tf
from step3_model import build_ctc_transformer_model
from step2_dataset import create_dataset_from_index_list, get_data_index_list, load_char2idx


class CTCModel(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

    def compile(self, optimizer):
        super().compile(optimizer=optimizer)
        self.optimizer = optimizer

    def call(self, inputs, training=False):
        # Forward pass wrapper: chuyển inputs qua base model
        return self.model(inputs, training=training)

    def train_step(self, data):
        features, labels, input_len, label_len = data
        labels = tf.cast(labels, tf.int32)
        input_len = tf.cast(tf.squeeze(input_len), tf.int32)
        label_len = tf.cast(tf.squeeze(label_len), tf.int32)

        with tf.GradientTape() as tape:
            logits = self.model(features, training=True)
            logits_time_major = tf.transpose(logits, [1, 0, 2])
            loss = tf.nn.ctc_loss(
                labels=labels,
                logits=logits_time_major,
                label_length=label_len,
                logit_length=input_len,
                logits_time_major=True,
                blank_index=-1,
            )
            loss = tf.reduce_mean(loss)

        tf.debugging.check_numerics(loss, "Loss is NaN hoặc Inf")

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 5.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        features, labels, input_len, label_len = data
        labels = tf.cast(labels, tf.int32)
        input_len = tf.cast(tf.squeeze(input_len), tf.int32)
        label_len = tf.cast(tf.squeeze(label_len), tf.int32)

        logits = self.model(features, training=False)
        logits_time_major = tf.transpose(logits, [1, 0, 2])
        loss = tf.nn.ctc_loss(
            labels=labels,
            logits=logits_time_major,
            label_length=label_len,
            logit_length=input_len,
            logits_time_major=True,
            blank_index=-1,
        )
        loss = tf.reduce_mean(loss)
        self.val_loss_tracker.update_state(loss)
        return {"val_loss": self.val_loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]


if __name__ == "__main__":
    batch_size = 32

    full_index_list = get_data_index_list("processed_chunks")
    total_samples = len(full_index_list)
    train_idx = full_index_list[: int(0.9 * total_samples)]
    val_idx = full_index_list[int(0.9 * total_samples):]

    train_dataset = create_dataset_from_index_list(train_idx, batch_size=batch_size)
    val_dataset = create_dataset_from_index_list(val_idx, batch_size=batch_size)

    char2idx = load_char2idx()
    vocab_size = len(char2idx)

    base_model = build_ctc_transformer_model(
        input_dim=128,
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        ff_dim=256,
    )

    ctc_model = CTCModel(base_model)

    # Build model bằng dummy input (bắt buộc cho lưu model)
    dummy_input = tf.random.uniform((1, 1000, 128))
    _ = ctc_model(dummy_input)

    ctc_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    os.makedirs("checkpoints", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/best_model.weights.h5",
            save_best_only=True,
            save_weights_only=True,  # Lưu full model
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
    ]

    steps_per_epoch = len(train_idx) // batch_size
    validation_steps = max(1, len(val_idx) // batch_size)

    ctc_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    # Lưu weights cuối cùng
    ctc_model.save_weights("checkpoints/final_model.weights.h5")
    print("✅ Đã lưu weights vào checkpoints/final_model.weights.h5")

    # Lưu mô hình full để export TFLite hoặc phục vụ inference sau này
    dummy_input = tf.random.uniform((1, 1000, 128))
    _ = ctc_model.model(dummy_input)
    
    # ✅ Lưu phần base model (để inference hoặc export TFLite sau này)
    export_dir = "exported/full_model"
    tf.saved_model.save(ctc_model.model, export_dir)
    print(f"✅ Đã lưu mô hình ở dạng SavedModel tại: {export_dir}")



