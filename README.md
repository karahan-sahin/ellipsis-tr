# Ellipsis in Turkish

## Overview

This project focuses on understanding the syntactic capabilities of large language models (LLMs) with a particular emphasis on the Turkish language. Our goal is to evaluate how well these models can identify and classify the types of elision in syntactic structures within given sentences. Additionally, we aim to assess their ability to recognize the prior context enabling the elision operation. To achieve these objectives, we have developed a comprehensive approach that spans novel data collection methods, model training, and a classification pipeline comprising three main tasks: Elliptical Type Classification, Discriminative Span Classification, and Extractive Span Classification.

## Models

### 1. Elliptical Type Classification: 

Classifies the type of ellipsis in a given sentence, helping to understand the syntactic operation.

```markdown
Ali eve geldi, ben gelmedim
# Output: Object Drop
```

### 2. Span Classification

#### 2.a Discriminative Span Classification: 

Identifies the specific span within a sentence that indicates the reason for the ellipsis.

```markdown
   Ali    onun         evine           gitti, ben gelmedim
#   O   B-Ellipsis   I-Ellipsis          O     O    O
```

#### 2.b Extractive Span Classification: 

Extracts the exact span that has been elided, providing insights into the syntactic structure and context.

```markdown
   Ali       onun              evine           gitti, ben gelmedim
#   O   <B-Object-Drop>   <I-Object-Drop>         O     O    O
```

## Getting Started

### Prerequisites
- Python 3.8+
- Pip
- transformers
- evaluate
- stanza
- nltk
- nltk


### Installation
Clone the repository:

```bash
git clone https://github.com/your-github/ellipsis-prediction-turkish.git
```

Navigate to the project directory:

```bash
cd ellipsis-prediction-turkish
```

(Optional) Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

- On Windows: `venv\Scripts\activate`
- On Unix or MacOS: `source venv/bin/activate`


Install the required dependencies:

```bash
pip3 install -r requirements.txt
```

## Usage

### 1. Training

#### a.Type Classification

```bash
python3 train_type_classification.py --model_name='' \
                                     --model_type='dbmbz/bert-turkish-cased-128k' \
                                     --epochs='20' \
                                     --augment=false \
                                     --batch_size='32' \
                                     --lr='1e-5' \
                                     --decay='1e-4' \
                                     --log_dir='' \
                                     --output_dir='' \
                                     --push_hub=false 
```

#### b. Span Classification

```bash
python3 train_span_classification.py --model_name='' \
                                     --model_type='dbmbz/bert-turkish-cased-128k' \
                                     --span_type='extractive' \
                                     --epochs='20' \
                                     --augment=false \
                                     --batch_size='32' \
                                     --lr='1e-5' \
                                     --decay='1e-4' \
                                     --log_dir='' \
                                     --output_dir='' \
                                     --push_hub=false 
```



### 2. Inference



#### 1. Elliptical Type Classification

Classify the type of ellipsis in a sentence.


```bash
python3 classify_ellipsis_type.py --sentence "Evde kim var?"
# Output: Ellipsis Type: Nominal
```

```python
from transformers import pipeline
from lib.processors import extractive_span_processor

span_clf = pipeline('<model-name>', preprocessor=extractive_span_processor)

span_clf("Ali onun evine gitti, ben gitmedim")

# Output: {
#   "elliptical_type": "Object Drop",
# }
```


#### 1. Discriminative Span Classification:

```bash
python3 classify_discriminative_span.py --sentence "Ahmet dün geldi, Mehmet ise..."
# Output: Discriminative Span: "Mehmet ise"
```

```python
from transformers import pipeline

span_clf = pipeline('<model-name>')

span_clf("Ali onun evine gitti, ben gitmedim")

# Output: {
#   "span": [ 'onun evine' ]
# }
```


#### 1. Extractive Span Classification:

```bash
python3 classify_extractive_span.py --sentence "Okula ben gittim, kardeşim ise evde kaldı."
# Output: Extractive Span: "evde kaldı"
```

```python
from transformers import pipeline
from lib.processors import extractive_span_processor

span_clf = pipeline('<model-name>', preprocessor=extractive_span_processor)

span_clf("Ali onun evine gitti, ben gitmedim")

# Output: {
#   "elliptical_type": "Object Drop",
#   "span": [ 'onun evine' ]
# }
```


## TODO List:

- [ ] Add OSCAR Corpus for data collection
- [ ] Add augment types
   - [ ] Add token replacement augmentation
   - [ ] Add backtranslation augmentation
- [ ] Add other pretrained models
- [ ] Add generative models

## Contributing
We welcome contributions from the community! Please refer to the CONTRIBUTING.md file for guidelines on how to contribute to this project.

## Citation

TBD

## License
This project is licensed under the MIT License - see the LICENSE file for details.