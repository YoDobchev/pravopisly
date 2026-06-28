# Pravopisly

![demo](./images/demo.gif)

Pravopisly is a WIP Bulgarian text correction project for detecting and correcting spelling, grammar, and punctuation errors.

## Setup

### 1. Install dependancies
```text
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the environment template:

```bash
cd model
cp .env-template .env
```

Open `.env` and update the following paths.

### 3. Download the [Bulgarian frequency list](https://huggingface.co/datasets/yodob/Bulgarian-Frequency-list):

Then set `FREQLISTPATH` to the downloaded frequency list path:

```env
FREQLISTPATH=/path/to/frequency-list
```

### 4. Detection Model

Download the [multi-head Bulgarian BERT](https://huggingface.co/yodob/Multi-Head-Bulgarian-BERT):

Then set `MHBERTPATH` to the downloaded model:

```env
MHBERTPATH=/path/to/multi-head-bert
```

### 5. Grammar Correction Model

Download the [mT5 Bulgarian grammar correction model](https://huggingface.co/yodob/mT5-Bulgarian-grammar-correction):

Then set `GRAMMARMODELPATH` in `.env` to the downloaded model path:

```env
GRAMMARMODELPATH=/path/to/mT5-Bulgarian-grammar-correction
```

### 6. Data (optional for training)

Download [dataset.jsonl](https://huggingface.co/datasets/yodob/Bulgarian-Text-Errors) and place it inside the `data` folder.


Set `DATAFOLDER` to the path of the `data` folder:

```env
DATAFOLDER=/path/to/pravopisly/data
```

## Running the Project

### 1. Start the gRPC Server

Run:

```bash
cd model
python grpc_server.py
```

### 2. Start the Go Server

Run:

```bash
cd be
go run src/main.go
```

### 3. Start the Frontend

Install dependencies:

```bash
cd view
npm install
```

Start the dev server:

```bash
npm run dev
```
