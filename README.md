# Pravopisly

Pravopisly is a WIP Bulgarian text correction project for detecting and correcting spelling, grammar, and punctuation errors.

## Setup

### 1. Data

Download `dataset.jsonl` and place it inside the `data` folder.
```text
https://huggingface.co/datasets/yodob/Bulgarian-Text-Errors/tree/main
```

### 2. Environment Configuration

Copy the environment template:

```bash
cp .env-template .env
```

Open `.env` and update the following paths.

Set `DATAFOLDER` to the path of the `data` folder:

```env
DATAFOLDER=/path/to/pravopisly/data
```

Download the Bulgarian frequency list from Hugging Face:

```text
https://huggingface.co/datasets/yodob/Bulgarian-Frequency-list/tree/main
```

Then set `FREQLISTPATH` to the downloaded frequency list path:

```env
FREQLISTPATH=/path/to/frequency-list
```

### 3. Detection Model

Install dependancies
```text
pip install -r requirements.txt
```

Download the multi-head Bulgarian BERT model from Hugging Face:

```text
https://huggingface.co/yodob/Multi-Head-Bulgarian-BERT/tree/main
```

Place the model files inside:

```text
model/checkpoints/pravopisly_model
```

Expected structure:

```text
pravopisly/
├── model/
│   └── checkpoints/
│       └── pravopisly_model/
```

### 4. Grammar Correction Model

Download the mT5 Bulgarian grammar correction model:

```text
https://huggingface.co/yodob/mT5-Bulgarian-grammar-correction/tree/main
```

Then set `GRAMMARMODELPATH` in `.env` to the downloaded model path:

```env
GRAMMARMODELPATH=/path/to/mT5-Bulgarian-grammar-correction
```

## Running the Project

### 1. Start the gRPC Server

Run:

```bash
python grpc_server.py
```

### 2. Start the Go Server

Run:

```bash
go run src/main.go
```

### 3. Start the Frontend

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```
