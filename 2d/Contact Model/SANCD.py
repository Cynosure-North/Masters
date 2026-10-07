import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

import BiGRU
import BERT
from data import GeometricDataset, pad_variable


class SANCD(nn.Module):
	def __init__(self, bigru_path, bert_path):
		super().__init__()

		self.bigru = BiGRU.instantiate_model(bigru_path)
		self.bert = BERT.instantiate_model(bert_path)

		self.threshold = 0.45

	def forward(self, data):
		_, output = self.forward_components(data)
		return output

	def forward_components(self, data):
		# BiGRU produces one vocabulary distribution per touch location.
		output_logits = self.bigru(data)

		top1_value, top1_predicted = nn.functional.softmax(output_logits, dim=-1).max(dim=-1)

		# Low-confidence predictions are replaced with the mask-token, which BERT fills in
		filtered_input = torch.where(
			top1_value > self.threshold,
			top1_predicted,
			torch.full_like(top1_predicted, BERT.CharTokenizer().mask_token_id),
		)

		output = self.bert(filtered_input)

		return output_logits, output



def _loss_for_logits(logits, labels, loss_fn):
	if logits.dim() == 3:
		logits = logits.reshape(-1, logits.size(-1))
	labels = labels.reshape(-1)
	return loss_fn(logits, labels)


def _set_requires_grad(module, enabled):
	for parameter in module.parameters():
		parameter.requires_grad_(enabled)


def train_model(
	model,
	dataloader,
	bert_epochs=3,
	bigru_epochs=3,
	alternating_epochs=3,
	gradient_clip=None,
	use_amp=True,
	checkpoint_path=None,
	validation_dataloader=None,
):
	"""Train BERT, then BiGRU, then alternate between both components."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
	bert_optimizer = torch.optim.AdamW(model.bert.parameters(), lr=1e-4)
	bigru_optimizer = torch.optim.Adam(model.bigru.parameters(), lr=1e-3)
	amp_enabled = use_amp and device.type == "cuda"
	scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

	def run_epoch(phase, epoch):
		model.train()
		_set_requires_grad(model.bert, phase == "bert")
		_set_requires_grad(model.bigru, phase == "bigru")
		optimizers = (bert_optimizer, bigru_optimizer)
		epoch_loss = 0.0
		batch_count = 0

		for batch_count, (labels, data) in enumerate(dataloader, start=1):
			data = data.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)
			active_index = 0 if phase == "bert" else 1
			optimizer = optimizers[active_index]
			optimizer.zero_grad(set_to_none=True)

			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
				output_logits, output = model.forward_components(data)
				logits = output if active_index == 0 else output_logits
				loss = _loss_for_logits(logits, labels, loss_fn)

			scaler.scale(loss).backward()
			if gradient_clip is not None:
				scaler.unscale_(optimizer)
				parameters = model.bert.parameters() if active_index == 0 else model.bigru.parameters()
				torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
			scaler.step(optimizer)
			scaler.update()
			epoch_loss += loss.detach().item()

			if active_index == 0:
				_set_requires_grad(model.bigru, False)
			else:
				_set_requires_grad(model.bert, False)

		if batch_count == 0:
			raise ValueError("The dataloader must contain at least one batch.")
			
		message = f"Epoch {epoch:<3.0f} -- {phase} loss: {epoch_loss / batch_count:.6f}"
		print(message)
		return epoch_loss / batch_count

	phase_schedule = (
		# Train the semantic component, then the geometric component, then alternate them.
		[("bert", epoch) for epoch in range(1, bert_epochs + 1)]
		+ [("bigru", epoch) for epoch in range(bert_epochs + 1, bert_epochs + bigru_epochs + 1)]
		+ [
			x		# The comprehension below require unfolding
			for xs in [(("bert", epoch), ("bigru", epoch+1)) for epoch in range(
				bert_epochs + bigru_epochs + 1,
				bert_epochs + bigru_epochs + (alternating_epochs*2) + 1,
				2)]
			for x in xs
		]
	)

	for phase, epoch in phase_schedule:
		run_epoch(phase, epoch)
		if validation_dataloader is not None:
			# Report both component losses after every phase epoch.
			validation_metrics = test_model(model, validation_dataloader, use_amp=use_amp)
			print(
				f"Validation -- BiGRU loss: {validation_metrics['bigru'][0]:.6f}, "
				f"BERT loss: {validation_metrics['bert'][0]:.6f}"
			)
		if checkpoint_path is not None:
			torch.save(model.state_dict(), checkpoint_path)

	_set_requires_grad(model.bert, True)
	_set_requires_grad(model.bigru, True)
	return model

@torch.no_grad()
def test_model(
	model,
	dataloader,
	use_amp=True,
):
	"""Evaluate SANCD and return loss and token accuracy for both components."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()
	loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
	amp_enabled = use_amp and device.type == "cuda"
	total_bigru_loss = 0.0
	total_bert_loss = 0.0
	total_bigru_correct = 0
	total_bert_correct = 0
	total_count = 0
	batch_count = 0

	for labels, data in dataloader:
		data = data.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
			bigru_logits, bert_logits = model.forward_components(data)

		bigru_loss = _loss_for_logits(bigru_logits, labels, loss_fn)
		bert_loss = _loss_for_logits(bert_logits, labels, loss_fn)
		bigru_predictions = bigru_logits.argmax(dim=-1)
		bert_predictions = bert_logits.argmax(dim=-1)
		valid_labels = labels.ne(-100)

		total_bigru_loss += bigru_loss.item()
		total_bert_loss += bert_loss.item()
		total_bigru_correct += ((bigru_predictions == labels) & valid_labels).sum().item()
		total_bert_correct += ((bert_predictions == labels) & valid_labels).sum().item()
		total_count += valid_labels.sum().item()
		batch_count += 1

	if batch_count == 0:
		raise ValueError("The dataloader must contain at least one batch.")

	return {
		"bigru": (total_bigru_loss / batch_count, total_bigru_correct / total_count if total_count else 0.0),
		"bert": (total_bert_loss / batch_count, total_bert_correct / total_count if total_count else 0.0),
	}


def main(_train_path=None, _validation_path=None, _test_path=None, _bigru_path=None, _save_path=None):
	project_dir = Path(__file__).resolve().parent
	data_dir = project_dir / "data" / "geometric"
	train_path = _train_path or data_dir / "train.csv"
	validation_path = _validation_path or data_dir / "validation.csv"
	test_path = _test_path or data_dir / "test.csv"
	bigru_path = _bigru_path or project_dir / "trained" / "best_BiGRU.pth"
	bert_path = project_dir / "trained" / "best_BERT.pth"
	save_path = _save_path or project_dir / "trained" / "best_SANCD.pth"

	model = SANCD(bigru_path=bigru_path, bert_path=bert_path)

	loader_kwargs = {
		"batch_size": 64,
		"collate_fn": pad_variable,
		"num_workers": 2,
		"pin_memory": True }
	train_dataloader = DataLoader(GeometricDataset(train_path), shuffle=True, **loader_kwargs)
	validation_dataloader = DataLoader(GeometricDataset(validation_path), shuffle=False, **loader_kwargs)
	print("data loaded")

	trained_model = train_model(
		model,
		train_dataloader,
		bert_epochs=3,
		bigru_epochs=3,
		alternating_epochs=3,
		gradient_clip=0.5,
		use_amp=True,
		checkpoint_path=save_path,
		validation_dataloader=validation_dataloader,
	)
	print("training complete")
	torch.save(trained_model.state_dict(), save_path)

	test_dataloader = DataLoader(GeometricDataset(test_path), shuffle=False, **loader_kwargs)
	metrics = test_model(trained_model, test_dataloader)
	for component, (loss, accuracy) in metrics.items():
		print(f"{component.title()} Test Loss: {loss:.6f}")
		print(f"{component.title()} Test Accuracy: {accuracy:.4%}")

if __name__ == "__main__":
	main()