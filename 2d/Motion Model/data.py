import gzip
import math
import queue
import random
import re
import threading
import zlib
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
import numpy as np
import torch

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
SEED = 0

class MotionDataset(torch.utils.data.Dataset):
	def __init__(self, path):
		self.path = Path(path)
		if not self.path.exists():
			raise FileNotFoundError(f"Motion dataset not found: {self.path}")

		self.files = sorted(self.path.glob("*.csv")) if self.path.is_dir() else None
		if self.files is None:
			raise ValueError(f"Motion dataset directory is empty: {self.path}")

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		file_path = self.files[idx]
		label = re.sub(r"_\d+$", "", file_path.stem).lower()
		try:
			features = np.loadtxt(file_path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
		except (OSError, ValueError) as exc:
			raise ValueError(f"Could not load numeric motion features from {file_path}") from exc

		if features.ndim != 2 or features.shape[0] == 0:
			raise ValueError(f"Motion record {idx} must contain at least one feature frame")

		# blank token is at the end of the alphabet
		targets = torch.tensor([chars.index(char) for char in label], dtype=torch.long)
		return torch.from_numpy(features), targets, len(targets)

####################################

class _ArticleTextParser(HTMLParser):
	"""Extract visible text from an article's HTML payload."""

	_IGNORED_TAGS = {"script", "style", "noscript", "svg"}

	def __init__(self):
		super().__init__(convert_charrefs=True)
		self.parts = []
		self._ignored_depth = 0
		self._ignored_tags = []

	def handle_starttag(self, tag, attrs):
		if tag in self._IGNORED_TAGS:
			self._ignored_depth += 1
			self._ignored_tags.append(tag)

	def handle_endtag(self, tag):
		if self._ignored_tags and tag == self._ignored_tags[-1]:
			self._ignored_tags.pop()
			self._ignored_depth -= 1

	def handle_data(self, data):
		if not self._ignored_depth:
			text = " ".join(data.split())
			if text:
				self.parts.append(text)


class CommonCrawlNewsStream:
	"""Stream a split-specific sample from recent Common Crawl News WARCs."""

	BASE_URL = "https://data.commoncrawl.org/crawl-data/CC-NEWS"
	SPLIT_MODULUS = 20
	SPLIT_BUCKETS = {"validation": set(range(3)), "test": set(range(3, 6))}

	def __init__(
		self,
		split,
		*,
		seed=SEED,
		max_warcs=16,
		lookback_months=12,
		documents_per_warc=100,
		min_chars=300,
	):
		if split not in {"train", "validation", "test"}:
			raise ValueError("split must be 'train', 'validation', or 'test'")
		if max_warcs < 1:
			raise ValueError("max_warcs must be at least 1")
		if lookback_months < 1 or documents_per_warc < 1:
			raise ValueError("lookback_months and documents_per_warc must be at least 1")
		self.split = split
		self.seed = seed
		self.max_warcs = max_warcs
		self.lookback_months = lookback_months
		self.documents_per_warc = documents_per_warc
		self.min_chars = min_chars
		self._warc_paths = None
		self._iteration = 0

	def _months(self):
		today = date.today()
		end_month = today.year * 12 + today.month - 1
		start_month = end_month - self.lookback_months + 1
		for month_index in range(start_month, end_month + 1):
			year, month = divmod(month_index, 12)
			yield year, month + 1

	@staticmethod
	def _configure_http_session(session):
		from requests.adapters import HTTPAdapter
		from urllib3.util.retry import Retry

		retry = Retry(
			total=5,
			connect=5,
			read=5,
			status=5,
			backoff_factor=1,
			status_forcelist=(429, 500, 502, 503, 504),
			allowed_methods=frozenset({"GET"}),
			respect_retry_after_header=True,
		)
		adapter = HTTPAdapter(max_retries=retry)
		session.mount("https://", adapter)
		session.mount("http://", adapter)

	def _get_warc_paths(self, *, refresh=False):
		if self._warc_paths is not None and not refresh:
			return self._warc_paths

		import requests

		paths = []
		with requests.Session() as session:
			self._configure_http_session(session)
			for year, month in self._months():
				url = f"{self.BASE_URL}/{year}/{month:02d}/warc.paths.gz"
				try:
					response = session.get(url, timeout=(15, 60))
					if response.status_code == 404:
						continue
					response.raise_for_status()
					listing = gzip.decompress(response.content).decode("utf-8")
				except requests.RequestException as exc:
					print(f"Skipping unavailable CC-NEWS listing {url}: {exc}")
					continue
				except (OSError, UnicodeDecodeError) as exc:
					print(f"Skipping invalid CC-NEWS listing {url}: {exc}")
					continue
				paths.extend(line.strip() for line in listing.splitlines() if line.strip())

		self._warc_paths = paths
		return paths

	def _selected_warc_paths(self, *, refresh=False):
		paths = []
		validation_buckets = self.SPLIT_BUCKETS["validation"]
		test_buckets = self.SPLIT_BUCKETS["test"]
		for path in self._get_warc_paths(refresh=refresh):
			bucket = zlib.crc32(path.encode("utf-8")) % self.SPLIT_MODULUS
			if self.split == "validation" and bucket in validation_buckets:
				paths.append(path)
			elif self.split == "test" and bucket in test_buckets:
				paths.append(path)
			elif self.split == "train" and bucket not in validation_buckets | test_buckets:
				paths.append(path)
		if not paths:
			raise FileNotFoundError(f"No Common Crawl News WARCs available for split {self.split!r}")

		cycle_seed = self.seed + self._iteration
		self._iteration += 1
		random.Random(cycle_seed).shuffle(paths)
		return paths[:self.max_warcs]

	@staticmethod
	def _extract_text(payload):
		parser = _ArticleTextParser()
		parser.feed(payload.decode("utf-8", errors="replace"))
		parser.close()
		return " ".join(parser.parts)

	def iter_documents(self, *, max_documents=None):
		"""Yield HTML text continuously, refreshing available WARC listings per cycle."""
		if max_documents is not None and max_documents < 1:
			return

		import requests

		documents_yielded = 0
		cycle = 0
		with requests.Session() as session:
			self._configure_http_session(session)
			while max_documents is None or documents_yielded < max_documents:
				paths = self._selected_warc_paths(refresh=cycle > 0)
				remaining = (
					max_documents - documents_yielded if max_documents is not None else None
				)
				per_warc_limit = self.documents_per_warc
				if remaining is not None:
					per_warc_limit = min(per_warc_limit, math.ceil(remaining / len(paths)))

				for path in paths:
					if max_documents is not None and documents_yielded >= max_documents:
						return
					url = f"https://data.commoncrawl.org/{path.lstrip('/')}"
					try:
						with session.get(url, stream=True, timeout=(15, 120)) as response:
							response.raise_for_status()
							with gzip.GzipFile(fileobj=response.raw, mode="rb") as archive:
								records_yielded = 0
								for warc_headers, payload in self._iter_warc_records(archive):
									if warc_headers.get("warc-type", "").lower() != "response":
										continue
									http_headers, body = self._parse_http_payload(payload)
									content_type = http_headers.get("content-type", "").lower()
									if content_type and "html" not in content_type:
										continue
									text = self._extract_text(body)
									if len(text) < self.min_chars:
										continue
									yield text
									documents_yielded += 1
									records_yielded += 1
									if records_yielded >= per_warc_limit:
										break
					except (requests.RequestException, OSError, EOFError, zlib.error) as exc:
						print(f"Skipping CC-NEWS WARC after retries failed: {path} ({exc})")
				cycle += 1

	@staticmethod
	def _iter_warc_records(stream):
		"""Read WARC records incrementally from a concatenated-gzip archive."""
		while True:
			line = stream.readline()
			while line and not line.startswith(b"WARC/"):
				line = stream.readline()
			if not line:
				return

			headers = {}
			while True:
				line = stream.readline()
				if not line or line in (b"\r\n", b"\n"):
					break
				name, separator, value = line.partition(b":")
				if separator:
					headers[name.decode("ascii", errors="ignore").lower()] = (
						value.decode("utf-8", errors="replace").strip()
					)

			try:
				content_length = int(headers["content-length"])
			except (KeyError, ValueError):
				return
			payload = stream.read(content_length)
			if len(payload) != content_length:
				return
			yield headers, payload

	@staticmethod
	def _parse_http_payload(payload):
		separator = b"\r\n\r\n"
		if separator not in payload:
			separator = b"\n\n"
			line_ending = b"\n"
		else:
			line_ending = b"\r\n"
		header_block, _, body = payload.partition(separator)
		http_headers = {}
		for line in header_block.split(line_ending)[1:]:
			name, separator, value = line.partition(b":")
			if separator:
				http_headers[name.decode("ascii", errors="ignore").lower()] = (
					value.decode("utf-8", errors="replace").strip()
				)
		if http_headers.get("content-encoding", "").lower() == "gzip":
			try:
				body = gzip.decompress(body)
			except OSError:
				pass
		return http_headers, body


class TextCorpus:
	"""A bounded in-memory text sample used for tokenizer fitting and evaluation."""

	def __init__(self, documents):
		self.documents = [str(document) for document in documents]
		if not self.documents:
			raise ValueError("TextCorpus cannot be empty")
		self.tokens = None

	def encode(self, tokenizer):
		"""Encode each document, delimiting documents with the tokenizer's BOS ID."""
		token_ids = [tokenizer.BOS_ID]
		for document in self.documents:
			token_ids.extend(tokenizer.encode(document))
			token_ids.append(tokenizer.BOS_ID)
		self.tokens = torch.tensor(token_ids, dtype=torch.long)
		return self.tokens

	def sample_batch(self, batch_size, block_size, device):
		if self.tokens is None:
			raise RuntimeError("Call encode(tokenizer) before requesting token batches")
		if self.tokens.numel() <= block_size:
			raise ValueError("Corpus is too short for the configured context window")

		starts = torch.randint(self.tokens.numel() - block_size, (batch_size,))
		inputs = torch.stack([
			self.tokens[start : start + block_size]
			for start in starts
		])
		targets = torch.stack([
			self.tokens[start + 1 : start + block_size + 1]
			for start in starts
		])
		return inputs.to(device), targets.to(device)

	def batches(self, block_size, device):
		"""Yield non-overlapping next-token batches for full-corpus evaluation."""
		if self.tokens is None:
			raise RuntimeError("Call encode(tokenizer) before requesting token batches")
		for start in range(0, self.tokens.numel() - 1, block_size):
			end = min(start + block_size + 1, self.tokens.numel())
			inputs = self.tokens[start : end - 1].unsqueeze(0).to(device)
			targets = self.tokens[start + 1 : end].to(device)
			if targets.numel():
				yield inputs, targets


class StreamingTextCorpus:
	"""Turn a remote document stream into prefetched sequential token batches."""

	def __init__(self, document_stream, tokenizer, *, max_documents=None, prefetch_batches=16):
		if prefetch_batches < 1:
			raise ValueError("prefetch_batches must be at least 1")
		self.document_stream = document_stream
		self.tokenizer = tokenizer
		self.max_documents = max_documents
		self._token_iterator = None
		self._buffer = []
		self._prefetch_batches = prefetch_batches
		self._batch_queue = queue.Queue(maxsize=prefetch_batches)
		self._stop_prefetch = threading.Event()
		self._prefetch_thread = None
		self._batch_config = None

	def _tokens(self):
		yield self.tokenizer.BOS_ID
		for document in self.document_stream.iter_documents(max_documents=self.max_documents):
			yield from self.tokenizer.encode(document)
			yield self.tokenizer.BOS_ID

	def _sample_cpu_batch(self, batch_size, block_size):
		if self._token_iterator is None:
			self._token_iterator = iter(self._tokens())

		sequence_length = block_size + 1
		needed = batch_size * sequence_length
		while len(self._buffer) < needed:
			try:
				self._buffer.append(next(self._token_iterator))
			except StopIteration as exc:
				raise ValueError("CC-NEWS stream ended before a training batch could be formed") from exc

		sequences = torch.tensor(self._buffer[:needed], dtype=torch.long).reshape(
			batch_size, sequence_length
		)
		self._buffer = self._buffer[needed:]
		return sequences[:, :-1], sequences[:, 1:]

	def _enqueue_prefetched(self, item):
		while not self._stop_prefetch.is_set():
			try:
				self._batch_queue.put(item, timeout=0.1)
				return True
			except queue.Full:
				continue
		return False

	def _prefetch_worker(self, batch_size, block_size):
		try:
			while not self._stop_prefetch.is_set():
				batch = self._sample_cpu_batch(batch_size, block_size)
				if not self._enqueue_prefetched(("batch", batch)):
					return
		except BaseException as exc:
			self._enqueue_prefetched(("error", exc))

	def sample_batch(self, batch_size, block_size, device):
		config = (batch_size, block_size)
		if self._prefetch_thread is None:
			self._batch_config = config
			self._prefetch_thread = threading.Thread(
				target=self._prefetch_worker,
				args=config,
				name="text-corpus-prefetch",
				daemon=True,
			)
			self._prefetch_thread.start()
		elif config != self._batch_config:
			raise ValueError("batch_size and block_size cannot change after prefetching starts")

		kind, result = self._batch_queue.get()
		if kind == "error":
			raise result
		inputs, targets = result
		return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

	def close(self):
		"""Stop the prefetch worker when training ends or is interrupted."""
		self._stop_prefetch.set()

####################################

def transform(data):
	data = np.asarray(data, dtype=float).reshape(2, 9, 3)

	# The final four markers per hand are Wout, Win, Aout, and Ain.
	wrist_positions = (data[:, 5] + data[:, 6]) / 2.0
	arm_positions = (data[:, 7] + data[:, 8]) / 2.0
	rotation_vectors = arm_positions - wrist_positions
	lengths = np.linalg.norm(rotation_vectors, axis=1, keepdims=True)
	wrist_rotations = rotation_vectors / lengths

	return np.hstack((data[:, :5], [wrist_positions, wrist_rotations]))
	



def preprocess(path):
	pass
	# TODO: preprocess for motion

if __name__ == "__main__":
	preprocess()