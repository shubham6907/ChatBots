# ChatBots
Simple Python code examples for different kind of chat bots.

### What is a chatbot?
A ChatBot is a kind of virtual assistant that can build conversations with human users! A Chatting Robot. Building a chatbot is one of the popular tasks in Natural Language Processing.

### Are all chatbots the same?
Chatbots fall under three common categories:
1. Rule-based chatbots
2. Retrieval-based chatbots
3. Intelligent chatbots

### Rule-based chatbots
These bots respond to users' inputs based on certain pre-specified rules. For instance, these rules can be defined as if-elif-else statements. While writing rules for these chatbots, it is important to expect all possible user inputs, else the bot may fail to answer properly. Hence, rule-based chatbots do not possess any cognitive skills.

### Retrieval-based chatbots
These bots respond to users' inputs by retrieving the most relevant information from the given text document. The most relevant information can be determined by Natural Language Processing with a scoring system such as cosine-similarity-score. Though these bots use NLP to do conversations, they lack cognitive skills to match a real human chatting companion.

### Intelligent AI chatbots
These bots respond to users' inputs after understanding the inputs, as humans do. These bots are trained with a Machine Learning Model on a large training dataset of human conversations. These bots are cognitive to match a human in conversing. Amazon's Alexa, Apple's Siri fall under this category. Further, most of these bots can make conversations based on the preceding chat texts.

## Python Libraries used for Development

### re
A Regular Expression or RegEx is a special sequence of characters that uses a search pattern to find a string or set of strings.
It can detect the presence or absence of a text by matching it with a particular pattern and also can split a pattern into one or more sub-patterns.

### numpy
NumPy is a general-purpose array-processing package. It provides a high-performance multidimensional array object and tools for working with these arrays. It is the fundamental package for scientific computing with Python. It is open-source software.

### pandas
Pandas is a powerful and versatile library that simplifies tasks of data manipulation in Python . Pandas is built on top of the NumPy library and is particularly well-suited for working with tabular data, such as spreadsheets or SQL tables. Its versatility and ease of use make it an essential tool for data analysts, scientists, and engineers working with structured data in Python.

### nltk
The Natural Language Toolkit (NLTK) is a Python package for natural language processing.

### sklearn
Scikit Learn or Sklearn is one of the most robust libraries for machine learning in Python. It is open source and built upon NumPy, SciPy, and Matplotlib. It provides a range of tools for machine learning and statistical modeling including dimensionality reduction, clustering, regression, and classification, through a consistent interface in Python. 
Additionally, it provides many other tools for evaluation, selection, model development, and data preprocessing. 

### spacy
spaCy is a library for advanced Natural Language Processing in Python and Cython. It's built on the very latest research, and was designed from day one to be used in real products.
spaCy comes with pretrained pipelines and currently supports tokenization and training for 70+ languages. It features state-of-the-art speed and neural network models for tagging, parsing, named entity recognition, text classification and more, multi-task learning with pretrained transformers like BERT, as well as a production-ready training system and easy model packaging, deployment and workflow management.

### tensorflow
TensorFlow is an open source software library for high performance numerical computation. Its flexible architecture allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices.

### transformers
Transformers provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio.

These models can be applied on:
📝 Text, for tasks like text classification, information extraction, question answering, summarization, translation, and text generation, in over 100 languages.
🖼️ Images, for tasks like image classification, object detection, and segmentation.
🗣️ Audio, for tasks like speech recognition and audio classification.

Transformer models can also perform tasks on several modalities combined, such as table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.
Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and can be modified to enable quick research experiments.
Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration between them. It's straightforward to train your models with one before loading them for inference with the other.

### torch
PyTorch is a Python package that provides two high-level features:
Tensor computation (like NumPy) with strong GPU acceleration
Deep neural networks built on a tape-based autograd system

## Other Important Concepts

### Tokenization
In the context of Natural Language Processing (NLP), a tokenizer is a crucial component responsible for breaking down a text into individual units, often words or subwords. These units are called tokens, and they serve as the fundamental building blocks for further NLP tasks.

Tokenization is a crucial preprocessing step in NLP, and the choice of tokenizer can significantly impact downstream tasks such as text classification, language modeling, machine translation, and more. Here are a few common types of tokenizers:

Word Tokenizer:
Splits the text into words based on space and punctuation.
Example: Input - "Tokenization is important." Output - ["Tokenization", "is", "important", "."]

Sentence Tokenizer:
Splits the text into sentences.
Example: Input - "Tokenization is important. It helps in NLP." Output - ["Tokenization is important.", "It helps in NLP."]

Subword Tokenizer:
Splits the text into subword units, which can be especially useful for handling rare or out-of-vocabulary words.
Example: Input - "Tokenization is important." Output - ["To", "ken", "iza", "tion", " is", " im", "port", "ant", "."]

Byte Pair Encoding (BPE) Tokenizer:
A type of subword tokenizer that iteratively replaces the most frequent pair of bytes (character sequences) with a new token.
Example: Input - "Tokenization is important." Output - ["To", "ken", "iz", "ation", " is", " imp", "ort", "ant", "."]

Tokenizer for Specialized Tasks:
Depending on the task, specialized tokenizers may be used. For instance, tokenizers for Named Entity Recognition (NER) may tokenize text in a way that preserves entities intact.

Pretrained Tokenizers:
Many modern NLP models, such as those based on transformer architectures, come with pretrained tokenizers tailored for specific models (e.g., BERT, GPT, etc.).

Tokenization is essential because it transforms unstructured text data into a format suitable for numerical processing, enabling the application of machine learning algorithms in NLP tasks. The choice of tokenizer often depends on the specific requirements of the task, language, and the characteristics of the text data.

### StopWords
In the context of Natural Language Processing (NLP), stopwords refer to words that are commonly used in a language but are generally considered to be of little value in terms of information content. These words are often filtered out during the preprocessing phase of text analysis because they are very common and don't carry significant meaning on their own.
Examples of stopwords in English include common words like "the," "is," "and," "in," "to," etc. These words are frequent in most documents and do not contribute much to the understanding of the content.

The process of removing stopwords from text is known as stopword removal. The main reasons for doing this include:
Noise Reduction: Stopwords often appear frequently in documents but don't provide meaningful information. Removing them can help reduce noise in the data.
Computational Efficiency: By eliminating stopwords, you reduce the amount of data that needs to be processed, making text analysis more computationally efficient.
Improved Model Performance: In many NLP tasks, including sentiment analysis, topic modeling, and document classification, removing stopwords can improve the performance of models by focusing on more relevant words.

Stopword lists can vary depending on the NLP task and the specific requirements of the analysis. Commonly used libraries, such as NLTK (Natural Language Toolkit) in Python, provide predefined lists of stopwords for various languages.

### Stemming
Stemming is a text normalization process in Natural Language Processing (NLP) that involves reducing words to their base or root form, called the "stem." The goal of stemming is to group words with similar meanings together, even if they have different inflections, prefixes, or suffixes. This can help in tasks like text analysis, information retrieval, and document categorization.

The process of stemming involves removing common prefixes or suffixes from words, keeping only the core part of the word that often represents its meaning. For example:
Original Word: jumping
Stem: jump
Original Word: running
Stem: run

Stemming is a heuristic process and may not always produce a valid word or the root word. Different stemmers use different algorithms and rules, and they may vary in terms of aggressiveness. One commonly used stemming algorithm is the Porter Stemmer.

### Vectorizer
In the context of Natural Language Processing (NLP), a vectorizer is a tool or process that converts raw text data into a numerical format, often in the form of vectors. The goal is to represent text data in a way that machine learning algorithms can understand and process it. There are different types of vectorizers used in NLP, and each has its own approach to converting text into numerical representations. 

Two common types of vectorizers are:
1. Count Vectorizer:
Method: It counts the frequency of each word in the document and represents the document as a vector of word counts.
2. TF-IDF Vectorizer (Term Frequency-Inverse Document Frequency):
Method: It considers both the frequency of a term in a document (term frequency) and the rarity of the term across all documents (inverse document frequency) to create a numerical representation.

These vectorization techniques are crucial in preparing text data for machine learning tasks, such as classification, clustering, or sentiment analysis, where numerical input is required. The choice of vectorizer depends on the specific task and the characteristics of the data.

### Cosine Similarity
Cosine similarity is a metric used to measure how similar two vectors are. In the context of Natural Language Processing (NLP), it is often employed to assess the similarity between two documents or text passages.

Applications in NLP:
Cosine similarity is frequently used in NLP for tasks such as document similarity, information retrieval, clustering, and recommendation systems.
It allows for a measure of similarity that is independent of the vector lengths, making it particularly useful when dealing with variable-length documents.

In summary, cosine similarity in NLP provides a way to quantify the similarity between two pieces of text by examining the cosine of the angle between their vector representations in a high-dimensional space.

### Sequence Padding
In the context of Natural Language Processing (NLP), sequence padding refers to the process of adding special tokens or placeholders to sequences of variable length to make them uniform in length. This is often done to facilitate the use of these sequences in machine learning models that require fixed-size input.

In NLP, text data is typically represented as sequences of words or tokens. However, since sentences or documents can vary in length, it becomes challenging to process them efficiently in machine learning models that expect fixed-size input.

Padding involves adding special tokens, often with a value of zero, to the shorter sequences so that they match the length of the longest sequence in the dataset. This ensures that all input sequences have the same length, allowing them to be fed into a neural network or other machine learning models that require fixed-size input.

### One-Hot Encoding
One-hot encoding is a technique used in natural language processing (NLP) and machine learning to represent categorical variables, including words or tokens in a text corpus. In the context of NLP, it is commonly used to convert words into a numerical format that can be easily fed into machine learning models.

Here's how one-hot encoding works:
Vocabulary Creation: First, a unique index is assigned to each unique word in the vocabulary of the corpus. This index essentially represents the position of the word in the vocabulary.
Binary Representation: For each word in a given text, a binary vector is created. The length of this vector is equal to the size of the vocabulary, and all elements are set to 0, except for the element at the index corresponding to the word's position in the vocabulary, which is set to 1.

For example, consider the following vocabulary:
["apple", "banana", "orange", "grape"]
One-hot encoding for the word "orange" would be:
[0, 0, 1, 0]
And for the word "banana":
[0, 1, 0, 0]

One-hot encoding has some advantages, such as simplicity and ease of implementation. However, it has limitations, especially when dealing with large vocabularies, as the resulting vectors can be very sparse and high-dimensional. Additionally, it does not capture semantic relationships between words.

Alternative approaches like word embeddings (e.g., Word2Vec, GloVe, or embeddings learned during neural network training) have been developed to address some of these limitations by representing words in dense, continuous vector spaces that capture semantic relationships.

### Text Corpus
In the context of Natural Language Processing (NLP), a text corpus refers to a large and structured set of text documents. It serves as a data source for training, evaluating, and testing various NLP models and algorithms. A text corpus can include a wide range of texts, such as books, articles, websites, or any other form of written or spoken language.

### Checkpoints
In the context of Natural Language Processing (NLP) and machine learning, a checkpoint typically refers to a saved copy of the model's parameters during training. It serves as a snapshot of the model's current state at a specific point in time. 

Checkpoints are useful for several reasons:
1. Resuming Training: If the training process is interrupted or stopped for some reason (e.g., hardware failure, manual intervention), you can resume training from the last saved checkpoint rather than starting from scratch. This is crucial for long and resource-intensive training processes.
2. Model Evaluation: Checkpoints allow you to evaluate the performance of the model on a validation set or test set at various points during training. This helps in monitoring the model's progress and selecting the best-performing version.
3. Fine-Tuning: You can use checkpoints to fine-tune a pre-trained model on a specific task. By loading the pre-trained weights, you can start training a related model with a head or additional layers for the specific task.
4. Experimentation: Checkpoints facilitate experimentation with different hyperparameters or model architectures. You can compare the performance of models trained with various configurations by loading their respective checkpoints.

In NLP, checkpoints are commonly used in training neural network models like recurrent neural networks (RNNs), long short-term memory networks (LSTMs), or transformer models for tasks such as language modeling, translation, sentiment analysis, and more. The checkpoints often include information about the model's architecture and learned parameters, allowing for easy reproducibility and sharing of models.
