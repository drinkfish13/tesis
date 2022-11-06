import torch
from catalyst import dl

class DBertRunner(dl.Runner):

    def predict_batch(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

    def handle_batch(self, batch):
        logits = self.predict_batch(batch)

        self.batch = {
            "logits": logits,
            "classes": torch.argmax(logits, dim=-1).long().reshape(-1),
            "labels": batch["labels"]
        }



