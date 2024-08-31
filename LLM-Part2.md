
# LLMs and their Fundamental Limitations (Part 2)

## Introduction

In [Part 1 of LLM and their Fundamental Limitations](https://hackmd.io/@LFNB9ifoT024aMHXU49sog/Bkh_RwLdC) we explored the idea that a language model, when trained on a comprehensive corpus of texts that encode relational descriptions, causal inference, prediction, and common sense, can effectively function as a world model. We discussed how such a model's performance scales with data size, adhering to a power law distribution—a pattern that naturally emerges from real-world interactions. Importantly, this analysis is implementation-agnostic, focusing on the theoretical underpinnings rather than specific architectures.

In this document, we delve into the specific challenges that large language models (LLMs) must overcome, the mathematical ideas behind potential solutions, and, most relevantly, how the Transformer architecture implements these solutions.

Before we begin, it is worthwhile to set some context.

A language model is a probabilistic framework designed to predict the next word in a sequence, given the preceding words. The goal is to assign a probability $P(w_t \mid w_1, w_2, \ldots, w_{t-1})$ to each possible next word $w_t$. Traditional n-gram models rely on the Markov assumption, which simplifies this task by considering only the last $n-1$ words. For instance, a trigram model estimates the probability of a word based on the two preceding words: $P(w_t \mid w_{t-2}, w_{t-1})$. Despite their simplicity, n-gram models suffer from *sparsity issues*—most potential n-grams are rarely seen in the training data—and *high-dimensionality*, as the number of possible n-grams grows exponentially with the size of the vocabulary $V$, leading to $V^3$ potential trigrams for $n=3$. This results in a large feature space that is difficult to manage and efficiently compute.

Starting from the seminal work of Bengio et al. [1], neural network-based language models address these limitations by using word embeddings, which represent words as dense vectors in a continuous space. Each word $w$ is mapped to a vector $e_w \in \mathbb{R}^d$, where $d$ is the embedding dimension (typically 50 to 300). These embeddings capture semantic similarities between words, allowing the model to generalize better. For example, words like "cat" and "dog" might have similar vectors, reflecting their semantic closeness. This compact representation reduces the dimensionality and sparsity issues inherent in n-gram models, enabling the neural network to process and learn from large amounts of text more effectively.

The architecture of language models has evolved significantly over time. Initial models used simple recurrent neural networks (RNNs) [2], which were later enhanced by Long Short-Term Memory networks (LSTMs) [3] and Gated Recurrent Units (GRUs) [4].

**The Transformer Wave**. The development of the Transformer architecture [5] has revolutionized language models, leading to significant advancements from BERT [6] to modern LLMs like GPT-3 [7]. Despite these advancements, the core architecture remains remarkably stable. As depicted in Figure 1, Transformer models process sequences of tokens, each represented as a high-dimensional vector (e.g., 12,288 dimensions in GPT-3). During inference, these models operate with a fixed context length (e.g., 2,000 tokens) and consist of multiple Transformer Blocks (96 in GPT-3), with each block's output feeding into the next.

![image](https://hackmd.io/_uploads/Syrn_radC.png)
**Figure 1**: Comprehensive view of the Transformer architecture, including tokenization, embedding, and the iterative nature of Transformer blocks.

The process begins with input text and proceeds through the following steps:

1. **Tokenization**: The input text is split into tokens, which are basic units of meaning (e.g., words or subwords).
2. **Embedding**: Each token is converted into a high-dimensional vector representation, capturing semantic information.
3. **Transformer Blocks**: The token sequence is passed through a series of Transformer blocks. Each block consists of an attention layer followed by a feed-forward layer, both employ residual connections, ensuring that the transformations are additive and preserve information learned during training:
   - **Attention Layer**: Injects correlations among tokens in the sequence, enabling the model to focus on relevant parts of the input.
   - **Feed-Forward Network (FFN) Layer**: Retrieves contextually relevant knowledge, processing each token independently after the attention operation.
4. **Output Token Selection**: The final layer at the current token position computes a probability distribution to select the next output token.
5. **Repetition**: The selected output token is fed back into the input, and the process repeats until the sequence is complete.

**The race of model size**. Transformer architecture [5] has led to the development of models like BERT [6], which achieved state-of-the-art performance at the time, with 110 million parameters for BERT Base and 340 million parameters for BERT Large. Modern large language models (LLMs) such as GPT-3 [7], which has 175 billion parameters, use very high feature dimensions (12,288) to handle vast amounts of data, providing them with the capacity to generate coherent and contextually relevant text. 


|	|BERT Large	|GPT-3	|
|---	|---	|---	|
|Parameters	|340 million	|175 billion	|
|Model Type	|Transformer-based encoder	|Transformer-based decoder-only	|
|Number of Layers	|24	|96	|
|Feature Dimensions	|1024	|12,288	|
|Number of Attention Heads	|16	|96	|
|Feed-Forward Network (FFN)	|1024 x 4096 x 1024 (ReLU)	|12,288 x 49,152 x 12,288 (ReLU)	|
|Layer Normalization	|Applied before each sub-layer and FFN	|Applied before each sub-layer and FFN	|
|Training Data	|Approx. 3.3 billion words (BooksCorpus and Wikipedia)	|570GB. Approx. 114 billion words (diverse internet text, including programming code) (see conversion note below)	|
|Training Objective	|Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)	|Causal Language Modeling	|
|Context Window	|Up to 512 tokens	|Up to 2048 tokens	|
|Primary Tasks	|Pre-training for NLP tasks such as question answering, text classification, named entity recognition	|Few-shot learning for a wide range of tasks including language translation, question answering, text generation, summarization	|
|Fine-Tuning	|Requires fine-tuning for specific tasks	|Capable of performing tasks with zero-shot, one-shot, and few-shot learning without fine-tuning	|
|Release Date	|2019	|2020	|
|Developed By	|Google	|OpenAI	|
|Key Strengths	|Strong performance on a variety of NLP benchmarks with fine-tuning	|High versatility and performance across diverse tasks without fine-tuning	|

There is the substantial increase in model and data size from BERT Large to GPT-3—approximately 515 times in model parameters and 35 times in training data. GPT-3's training data is reported as around 570GB of diverse internet text. To estimate the number of words, consider that an average word consists of 5 characters (including spaces and punctuation), with each character typically occupying 1 byte. Using this approximation, the number of words in GPT-3's training data can be estimated as follows:

$$
\text{Estimated Words} = \frac{570 \times 10^9 \text{ bytes}}{5 \text{ characters/word}} = \text{114 billion words}
$$

We will mostly concern ourselves on pretrained LLMs, using GPT-3 as the primary example, and with next-token prediction as the only training objective. For completeness, here are two most notable techniques after GPT-3, SFT and RLHF:
- **Supervised Fine-Tuning (SFT)**: GPT-3 is primarily trained in an unsupervised manner on a diverse corpus of text, allowing it to generalize and perform a wide range of tasks with minimal examples provided at inference time (few-shot learning). However, GPT-3 and its successors, such as InstructGPT, can also be fine-tuned using Supervised Fine-Tuning (SFT) for specific tasks to improve performance further. SFT involves training the model on labeled datasets for particular applications. Major tasks for SFT include language translation, sentiment analysis, text classification, and more [8].
- **Reinforcement Learning from Human Feedback (RLHF)**: Reinforcement Learning from Human Feedback (RLHF) is another technique introduced in models like InstructGPT and ChatGPT to improve alignment with human preferences [9]. This method involves using human evaluators to provide feedback on the model's outputs, which is then used to optimize the model's performance through reinforcement learning.

**Roadmap** We will start with [Section 2](#Key-Challenges-and-How-Transformer-Handles-Them), which will highlight the core challenges that *any* large language model (LLM) implementation must address. Other than [efficient vocabulary representation](#C1:-Efficient-Vocabulary-Representation), which is not unique to Transformer, the subsequent sections will delve into each of these challenges in turn, along with the mechanisms the Transformer adopts to tackle them: [learns and samples from distributions contextually](#Learning-and-Sampling-from-Distributions), [manages the curse of dimensionality](#Handling-the-Curse-of-Dimensionality), [captures diverse relationships](#Capturing-Diverse-Relationships), and [processes out-of-prompt knowledge](#Handling-Out-of-Prompt-Knowledge), all within a framework of [iterative refinement and hierarchical learning](#Iterative-Refinement-and-Hierarchical-Learning). Finally, one of the most intriguing aspects of LLMs is the emergent behavior observed after next-token prediction-based pretraining. We will explore this phenomenon by discussing [In-Context Learning (ICL)](#In-Context-Learning) and [Chain-of-Thought (CoT) reasoning](#Chain-of-Thought-(CoT)-Reasoning) in [Emerging Behavior of LLMs](#Emerging-Behavior-of-LLMs).

<a name="key-challenges-and-how-transformer-handles-them"></a>
## Key Challenges and How Transformer Handles Them
This section describes the key challenges that **any** LLMs at a very large scale have to deal with, and how Transformers handle them. To begin, any external signals, texts, or visuals, must be converted into tokens of embedding—vectors whose contents are learnable during training (a vector of 12,288 floats in GPT-3). This process involves breaking down complex inputs into manageable representations, which introduces the first challenge: efficiently representing a vast and diverse vocabulary. **[Efficient Vocabulary Representation]**

Any generative model must be able to learn a complex distribution to generate new samples, and the distribution is contextual, meaning it depends heavily on the surrounding information or context. The challenge here is to accurately learn and sample from these distributions, ensuring that the generated text or output is coherent and contextually appropriate. Transformers achieve this through mechanisms similar to classical kernel smoothing, while also incorporating methods to respect the order and position of words. **[Learning and Sampling from Distributions]**

Learning contextual distributions requires understanding how tokens relate to one another within a given context. Additionally, larger tokens (i.e., why not 500,000 instead of 12,288?) can lead to more model capacity. However, in high-dimensional spaces, data points become sparse, and distance metrics become less informative. These issues, known as the curse of dimensionality, make it harder for the model to differentiate between relevant and irrelevant features. Transformers address this by implementing approximations that capture the core structure of the data. **[Handling the Curse of Dimensionality]**

Many tasks require knowledge beyond the immediate input context. Thus, the ability to handle out-of-prompt information is crucial. Transformers tackle this challenge with feed-forward networks (FFNs), which store and retrieve relevant knowledge, acting as a dynamic memory system that allows the model to generalize beyond the input prompt. **[Handling Out-of-Prompt Knowledge]**

To process complex signals such as text, the model at each step learns correlations among inputs (attention) and then retrieves and applies the necessary knowledge (FFN). The output is passed to the next layer, where this process is repeated, progressively merging simple patterns into more complex ones and eventually computing a density function to generate new tokens. The stacked architecture of Transformers can be thought of as a loop of iterations—up to 96 in GPT-3—each with different parameters, allowing for continuous refinement and deeper learning. **[Iterative Refinement and Hierarchical Learning]**

### C1: Efficient Vocabulary Representation

Language models must efficiently handle a vast vocabulary, ranging from common words like "the" to rare terms like "antidisestablishmentarianism." The most commonly used English words number only in the thousands—some estimates suggest that just 3,000 words make up about 95% of everyday communication. However, there are more than a million words in the English language overall, including scientific terms, jargon, and other specialized vocabulary. This expansive range follows a power law distribution, where a small number of words are used frequently, while the majority of words are rare. Researchers have estimated that the total number of words in English could be around 1,022,000, with new words being added continuously as language evolves, reflecting the language's adaptability to cultural and technological changes [(EF English Live)](https://englishlive.ef.com/en/blog/language-lab/many-words-english-language/), [(Merriam-Webster)](https://www.merriam-webster.com/help/faq-how-many-english-words).

This dynamic and growing vocabulary poses a significant challenge for language models, as they must efficiently represent both common and rare words, often within a single model.

**Transition to Subword Tokenization**: To address these limitations, Byte Pair Encoding (BPE) was introduced as a solution. BPE is fundamentally a compression algorithm that leverages the statistical properties of subwords to efficiently represent language. It starts with an initial vocabulary of individual characters and iteratively merges the most frequent pairs of symbols or subwords to form new subwords.

For instance, in "antidisestablishmentarianism," BPE would break the word down into subwords like "anti," "dis," "establish," "ment," "arian," and "ism." These subwords, being more frequent across different contexts, allow the model to efficiently represent the original word, reducing the vocabulary size while enhancing its ability to generalize across diverse linguistic expressions. While the power law distribution still applies to subwords, this decomposition mitigates the problem by ensuring that even rare words can be represented by more common, reusable subword units, thus smoothing the distribution and making it more manageable for the model.

**Embedding and Usage**: After tokenization into subwords, each subword is mapped to a dense vector, or embedding, stored in an embedding table. The model retrieves the corresponding embedding vectors and uses them as inputs for subsequent processing layers, such as Transformer layers. These embeddings are dynamically adjusted during training, allowing the model to refine its understanding and generation of text based on the context in which the subwords appear.

### C2: Learning and Sampling from Distributions

In generative modeling, the ability to produce coherent and contextually appropriate text hinges on the model’s capacity to accurately learn and sample from complex distributions. Language models must learn these distributions directly from data, adapting to the nuances of the input text. The core challenge is that the distribution from which the next word should be drawn is not predefined. Instead, it must be learned from the data, with the model dynamically adapting based on the context provided by preceding words.

**Kernel Smoothing and Alternatives**: One way to understand how Transformers achieve this is through the concept of kernel smoothing. In statistics, kernel smoothing is a non-parametric method for estimating the probability distribution of a variable by weighting nearby observations. Similarly, in a Transformer, the model can be seen as performing a form of kernel smoothing, where it weighs the influence of each word in the context based on its relevance or "distance" to the predicted word.

For example, consider the sentence "The cat quickly chased the mouse." To predict the next word after "chased," the model needs to determine which words in the sentence are most relevant. Here’s how the Transformer might handle this:
- **Semantic Relationships**: Words like "cat" and "mouse" are semantically related to the verb "chased," and the attention mechanism will likely assign them higher attention scores. This is akin to kernel smoothing, where nearby observations (in this case, semantically related words) are weighted more heavily. This influences the probability of picking "mouse" as the target word.
- **Positional Embeddings**: The position of each word in the sentence is crucial. The model needs to know that "quickly" modifies "chased," and "the cat" is the subject, while "the mouse" is the object. Positional embeddings provide this structural information, helping the model maintain a coherent lexical structure in its predictions.
- **Respecting Causal Order**: In generating text, the Transformer must also respect the order of words. For instance, when predicting the word after "chased," the model uses a causal mask to ensure that previous tokens influence the prediction, but not vice versa—maintaining the sequential integrity of the sentence. In our example, "chased" influences the prediction of "the," and subsequently, "the" influences the prediction of "mouse," but "mouse" does not affect the prediction of earlier tokens.

While kernel smoothing provides a flexible framework for learning distributions, it is computationally intensive, especially in high-dimensional spaces like those encountered in language models. The efficiency in Transformers comes from the fact that the projection to lower-dimensional spaces is not only pre-learned but also specifically optimized to capture the most relevant information. How to handle the curse of dimensionality is what we will discuss next.

### C3: Handling the Curse of Dimensionality

Language models like GPT-3 operate with an embedding dimension of 12,288—over 40 times larger than the typical 300 dimensions used in early embedding models like Word2Vec and GloVe. While this high dimensionality allows for a richer representation of tokens, capturing more nuanced semantic and syntactic features, it also introduces significant challenges related to computational efficiency and the curse of dimensionality.

**Curse of Dimensionality**: The curse of dimensionality refers to the phenomenon where the volume of space increases so rapidly with the number of dimensions that data points become sparse. In high-dimensional spaces, the concept of distance becomes less informative, making it difficult to meaningfully differentiate between data points. This sparsity can hinder the model's ability to generalize from the training data, leading to inefficiencies and potential overfitting.

![image](https://hackmd.io/_uploads/B1MpMlaqA.png)

**Figure**: This chart illustrates two key aspects of the curse of dimensionality. The blue line (left y-axis) shows the exponential growth in the volume of a hypersphere as dimensions increase, calculated as $2^d$. The green line (right y-axis) demonstrates how the proportion of space considered "near" (within 1% of the edge of a unit hypercube) rapidly decreases, calculated as $0.99^d$. These trends highlight why high-dimensional spaces pose challenges for machine learning algorithms, including those used in language models.

**Low-Rank Approximation and Dimensionality Reduction**: Consider a sequence of tokens represented by an $n$-by-$d_x$ matrix $X$, where $n$ is the number of tokens, and each token is described by a $d_x$-dimensional feature vector (e.g., 12,288 in GPT-3). To capture the relationships between tokens, we can compute the matrix $X X^T$, which shows how each token correlates with others. However, this approach has two major drawbacks: 1) while $X$ as an embedding can be learned, its capacity to adapt contextually is limited, and 2) it quickly encounters the curse of dimensionality, making computation infeasible in high-dimensional spaces.

To address these issues, Transformer introduces a learnable matrix $A$ and instead compute $X A X^T$. In addition, to deal with the challenge of high-dimensionality, instead of learning a large, $d_x$-by-$d_x$ matrix, $A$ is decomposed into the product of two tall and skinny matrices, significantly reducing the dimensionality. This is the core idea of the attention mechanism.

In the extreme case, these matrices could be reduced to two vectors of dimension $d_x$ (i.e., 12,288 elements in GPT-3). However, GPT-3 actually uses 128 pairs of vectors, achieving a rank-128 approximation. This still represents a highly aggressive form of dimensionality reduction, reducing the dimensionality to about 1% of the original feature size, while retaining the capacity to capture rich, nuanced interactions between tokens. 

**PCA Analogy**: A helpful way to understand low-rank approximation is through Principal Component Analysis (PCA). PCA reduces dimensionality by identifying the most significant directions, or principal components, in which the data varies. Similarly, in Transformers, low-rank approximation forces the model to focus on the most critical interactions between tokens. By decomposing the matrix $A$ into two tall and skinny matrices, the model's parameters are trained to capture the most informative parts of the data, ensuring efficient and contextually appropriate text generation.

### C4: Capturing Diverse Relationships

One of the challenges in language modeling is the need to capture and understand diverse relationships within the input text. These relationships can be syntactic, semantic, positional, or contextual, and a model's ability to grasp these nuances significantly impacts its performance. Traditional models often struggled to balance these aspects, either focusing too much on one type of relationship (e.g., syntax) or being unable to generalize across different contexts.

**Multi-Head Attention & Fractional Learning**: Transformers address this challenge through the mechanism of multi-head attention, which allows the model to capture various types of relationships simultaneously. Multi-head attention enables the model to apply multiple "heads," or parallel attention mechanisms, each focusing on different aspects of the input data. This process is an application of the broader concept of fractional learning, where a complex task is divided into smaller, more manageable parts, each of which can be learned independently.

Consider our example sentence, "The cat quickly chased the mouse." In this context, one attention head might focus on the syntactic relationship between "cat" and "chased," another might focus on the adverbial modifier "quickly," and yet another might consider the object "mouse." Each head, by operating on a different aspect of the sentence, contributes to a more nuanced and comprehensive understanding of the input.

**Interaction Between Heads**: The outputs from all the attention heads are then combined to form a single, unified representation of the input. This combination is typically done by concatenating the outputs of the heads, which are then processed further, often by the Feed-Forward Network (FFN) that follows next. While each head operates independently, the final combination allows the model to integrate the diverse relationships captured by each head, leading to a richer representation of the input.

In Transformers, the effectiveness of multi-head attention depends on the balance between overlap and diversity among the heads. If the heads are too similar, they may redundantly focus on the same features, reducing the model's capacity to generalize. Conversely, if the heads are too disjoint, they might fail to capture essential overlaps or interactions between features necessary for a holistic understanding of the input. The combination of heads in subsequent layers allows for more complex interactions and refinements.


### C5: Handling Out-of-Prompt Knowledge

In many tasks, language models need to go beyond the information explicitly provided in the input prompt. This ability to handle *out-of-prompt* knowledge is crucial for answering questions that require external knowledge or making inferences based on world knowledge not present in the immediate context.

**Feed-Forward Networks (FFN) & Non-Linearity**: To address out-of-prompt needs, Transformers employ Feed-Forward Networks (FFN) following the attention mechanism in each layer. The FFN acts as a powerful tool for associating, storing, and retrieving relevant knowledge that may not be directly provided in the input sequence.

Consider our earlier example sentence, "The cat quickly chased the mouse." After the attention mechanism has processed this input, the FFN takes the output and applies a transformation that allows the model to relate this sequence to broader, contextual knowledge, often required for commonsense reasoning. For instance, the FFN might prepare the context that something dramatic is about to happen.

There are a few important features of the FFN, which is structured as a two-layer MLP (Multilayer Perceptron) with a widened middle layer and ReLU non-linearity, i.e. an expand-compress architecture:
- **KV-store**: This design expands the input into a higher-dimensional space, activating different neurons (features) to retrieve relevant knowledge, and then compresses the output back to the original dimensionality with enriched content, reflecting a deeper understanding of the input.
- **Feature Disentangling**: This expand-compress architecture also facilitates the disentangling of complex and overlapping features, which is critical for tasks where the input may involve multiple, intertwined concepts.

The expand-compress configuration aligns with the Universal Approximation Theory [15], which suggests that a sufficiently wide feed-forward network can approximate any continuous function. While FFNs are central to storing and processing knowledge, it is important to note that knowledge can also be distributed across other model parameters, including the weights of the attention layers and embeddings.


### C6: Iterative Refinement and Hierarchical Learning

To handle the complexity of language, Transformers utilize a deep architecture where each layer builds upon the previous one. This process of **iterative refinement** allows the model to progressively develop more abstract and comprehensive representations of the input data.

**Comparison with BPE**: In Byte Pair Encoding (BPE), the merging of subwords follows a fixed, pre-determined process based on statistics from a static corpus. Once the BPE dictionary is created, it remains unchanged during model operation. Transformers, however, engage in a dynamic and ongoing process of refinement as they stack multiple layers. With each layer, the model refines its understanding of the input data and the knowledge stored in the Feed-Forward Networks (FFNs).

**Dynamic Refinement Across Layers**: Each layer in a Transformer re-evaluates and adjusts the relationships between tokens identified in the previous layer. This dynamic refinement is akin to a real-time, iterative merging process, where the model continuously integrates and optimizes its understanding of the input sequence. This process allows the model to capture increasingly complex patterns and hierarchical structures in the data.

For instance, revisiting our earlier example, in the sentence "The cat quickly chased the mouse," the first layer might focus on identifying basic relationships like "cat" and "chased." In the second layer, the model might further refine this understanding by incorporating the adverb "quickly," leading to a more nuanced interpretation of the action. This hierarchical learning enables the model to build deeper, more sophisticated representations as it processes more layers.

**Skip Connections**: Skip connections, or residual connections, are a crucial component in enabling this deep and iterative refinement. They are applied around both the attention mechanism and the Feed-Forward Network (FFN) within each layer. These connections allow the original input to each layer to be added back to the output, helping to stabilize the training process. They prevent the model from losing important information as it passes through multiple layers, making it possible to build very deep models without suffering from issues like vanishing gradients.

### Putting it all Together
![image](https://hackmd.io/_uploads/SJn4XK35R.png)


We've covered the core concepts behind the Transformer architecture, but two more critical aspects deserve attention:
- **Normalization**: Before applying the attention mechanism and the Feed-Forward Networks (FFN), a Layer Normalization stage ensures stability and efficiency during training. By normalizing inputs across features, it prevents internal covariate shift and aids in faster convergence. Normalization is also embedded elsewhere in the Transformer, such as the scaling factor of the square root of the feature size in the attention mechanism, which helps control the magnitude of dot products.
- **Parallelism**: The attention mechanism functions fundamentally as a set operation, meaning it applies uniformly across all elements in the context. Therefore most of the operations in the attention sub-layer is parallel, except the softmax operator rescales correlations and introduces $\text{O}(L^2)$ complexity (with $L$ being the context length), all computations are matrix operations, leveraging the power of modern hardware for efficient parallel processing.

The following table summarizes the key challenges addressed by the Transformer architecture, the corresponding mathematical solutions, and how these are implemented within the model:

| **Challenge**                                        | **Mathematical Concepts/Implementation**           |
|------------------------------------------------------|----------------------------------------------------|
| Efficient Vocabulary Representation                  | **Statistical Compression (e.g., BPE):** Subword tokenization using BPE; Combats vocabulary power law distribution |
| Learning and Sampling from Distributions             | **Kernel Smoothing & Contextual Computation:** Attention mechanism adjusts weights; Positional embeddings add structure; Causal masking ensures lexical causality |
| Handling the Curse of Dimensionality                 | **Low-Rank Approximation & Dimensionality Reduction:** Low-rank approximations in attention; Reduces computational complexity; Captures essential token interactions |
| Capturing Diverse Relationships                      | **Fractional Learning & Multi-Channel Processing:** Multi-head attention enables parallel processing; Heads focus on different aspects of the input |
| Handling Out-of-Prompt Knowledge                     | **Multi-Layer Feed-Forward Networks (FFN) & Non-Linearity:** Two-layer MLPs enable universal approximation; FFNs act as associative memory; Non-linearity aids in feature disentangling |
| Iterative Refinement and Hierarchical Learning       | **Layer Stacking & Iterative Refinement:** Dynamic refinement across layers; Each layer builds on previous representations |
| Extensive Parallelizable Operations                  | **Matrix Operations: Cross-Token and In-Token:** Cross-token operations in attention; In-token operations in MLP layers; All operations parallelizable except softmax ($\text{O}(L^2)$ over context length of $L$) |

This table encapsulates how Transformers, through a combination of statistical methods, mathematical approximations, and deep learning techniques, overcome the many challenges posed by natural language processing tasks. The architecture's modularity and depth allow it to scale efficiently, handle complex patterns, and generalize well across a wide range of contexts and tasks.

<a name="learning-and-sampling-from-distributions"></a>
## Learning and Sampling from Distributions

Earlier in the Key Challenges section, we explored the critical task of learning complex distributions from data, a fundamental capability for any language model. This section delves into how the Transformer architecture addresses these challenges through three key mechanisms: first, by employing kernel smoothing and its alternatives to manage distributional learning; second, by using positional embeddings to preserve sequence order; and finally, by enforcing causal order during text generation. These components are essential to the Transformer's ability to effectively model and generate coherent language.

### Kernel Smoothing and Alternatives

**Understanding Kernel Smoothing**:  

![image](https://hackmd.io/_uploads/Sk384sUOA.png)  
*Example of kernel smoothing: Each data point is updated based on samples close by, and we can estimate a smooth curve.*  
(from https://www.walker-harrison.com/posts/2021-02-13-visualizing-how-a-kernel-draws-a-smooth-line/)

Kernel smoothing is a non-parametric method used in statistics to estimate the probability distribution or density function of a variable by averaging nearby observations. Traditionally, this technique is used to estimate a function $y = f(x)$, where the goal is to smooth the target values $y$ based on the input features $x$. Once this smoothed estimate $\hat{f}(x_i)$ is obtained, it reflects the underlying distribution of the data points, allowing new samples to be drawn, which represent possible outcomes based on the learned distribution.

However, kernel smoothing can also be applied directly to the input features $x$ themselves, aiming to estimate a new, refined representation $x' = f(x)$. This approach is particularly relevant in contexts like the Transformer architecture, where the goal is to refine the features to better capture the underlying distribution of the data.

The influence of each observation on the estimate at a given point is determined by a weight, referred to as the influence weight, denoted as $\alpha_{i,j}$. This weight is computed using a kernel function $K(x_i, x_j)$, which quantifies the similarity or proximity between points $x_i$ and $x_j$.

Mathematically, for a set of observations $X = \{x_1, x_2, \ldots, x_n\}$, the smoothed or refined estimate at a point $x_i$ can be expressed as a weighted average of the surrounding data points:

$$
x_i' = \sum_{j=1}^{n} \alpha_{i,j} \cdot x_j
$$

Here, the influence weight $\alpha_{i,j}$ is derived from the kernel function $K(x_i, x_j)$, ensuring that observations closer to $x_i$ have a greater impact on the refined estimate:

$$
\alpha_{i,j} = \frac{K(x_i, x_j)}{\sum_{j=1}^{n} K(x_i, x_j)}
$$

A common choice for the kernel function is the Gaussian kernel, which emphasizes observations near $x_i$:

$$
K(x_i, x_j) = \exp\left(-\frac{(x_i - x_j)^2}{2\sigma^2}\right)
$$

In the context of Transformers, this smoothing process directly refines the input features $x$ into a new set of features $x'$. This transformation is captured mathematically as:

$$
X' = \alpha X
$$

This formulation represents a form of kernel smoothing, where each data point is adjusted based on a weighted average of other points, with the weights reflecting their similarity. However, instead of using a Gaussian kernel, the Transformer employs a measure of similarity derived from learned matrices, which we will explain next.

**Implementing Kernel Smoothing in Transformers**:  
In the context of Transformers, the attention mechanism can be viewed as an implementation of kernel smoothing, where the influence weight $\alpha_{i,j}$ is used to determine the relevance of each word in a sequence relative to others. Instead of estimating the distribution of a single variable, the Transformer uses the attention mechanism to estimate the relevance of each word in a sequence, effectively learning and sampling from a complex distribution of words in a high-dimensional space.

The Transformer achieves this by projecting the input sequence into three different spaces: queries ($Q$), keys ($K$), and values ($V$). These projections are obtained through learned weight matrices:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

Here, $X$ is the input sequence, and $W_Q$, $W_K$, and $W_V$ are the weight matrices that map the input into the query, key, and value spaces, respectively. This entire operation can be summarized as a function, in the case of GPT3, mapping the feature vector of 12,288 dimensions to a subspace of dimension 128, $\text{ATTN} : \mathbb{R}^{d_x} \to \mathbb{R}^{d_k}$.
:

$$
\text{ATTN}(X) = \text{softmax}\left(\frac{XW_Q (XW_K)^T}{\sqrt{d_k}}\right) XW_V
$$

The attention mechanism computes the relevance or attention scores by taking the dot product of the queries and keys, scaled by the square root of the key dimension $d_k$. These attention scores are directly analogous to the influence weights $\alpha_{i,j}$ in kernel smoothing:

$$
\alpha_{i,j} = \frac{\exp\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)}{\sum_{j'} \exp\left(\frac{q_i \cdot k_{j'}}{\sqrt{d_k}}\right)}
$$

In this framework, $\alpha_{i,j}$ represents how much influence or attention word $x_j$ should have on word $x_i$. The values $V$ are then weighted by these influence scores to compute the output:

$$
V' = \sum_{j=1}^{n} \alpha_{i,j} \cdot v_j = \alpha V
$$


**A few points worth mentioning**: 
- The attention mechanism in Transformers can be viewed as a form of kernel smoothing applied directly to the input features. The attention scores $\alpha_{i,j}$ function similarly to influence weights in kernel smoothing, determining the degree to which each word in the sequence influences the others.
- The refinement occurs on a linear projection of $X$ into a lower-dimensional space $V$, resulting in $V'$. However, the smoothing process is independent of this projection—$W_V$ could be an identity matrix, and the smoothing would still operate effectively.
- Unlike traditional non-parametric kernel smoothing, the Transformer’s attention mechanism is parametric, with learned projections into query, key, and value spaces. In addition, unlike many kernel methods, the attention mechanism in Transformers is not symmetric; the influence of a word on another is not necessarily the same in reverse. This parametric approach allows for efficient low-rank approximations, enabling the model to capture and compute complex and directional relationships within the data more effectively. 

### Positional Embeddings

While the attention mechanism allows the Transformer to focus on relevant parts of the input, it inherently treats input tokens as a set, ignoring their order. This limitation arises because the attention mechanism does not consider the position of words in a sequence, which is crucial for understanding language. To address this, the Transformer introduces positional embeddings, which encode the position of each token in the sequence, providing the necessary information about word order.

**Absolute Positional Embeddings**:  
Positional embeddings are typically derived using sine and cosine functions of varying frequencies, ensuring that each position has a unique embedding while maintaining a smooth representation over the sequence. For an input sequence of length $n$ and model dimension $d_{\text{model}}$, the positional embedding for a token at position $pos$ and embedding dimension $i$ is computed as:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

These embeddings are added to the input token embeddings:

$$
x_i^{\text{pos}} = x_i + PE_i
$$

This approach allows the model to distinguish between tokens based on their positions, ensuring that word order is respected during text generation. By combining positional embeddings with the attention mechanism, the Transformer can capture and sample from distributions that reflect the sequential dependencies in the data.

**Relative Positional Embeddings**:  
In addition to absolute positional embeddings, some Transformer variants use relative positional embeddings, such as Rotary Position Embedding (RoPE). Relative positional embeddings encode the relative distances and directions between tokens, which can be particularly useful for tasks where the relative positioning of tokens is more significant than their absolute positions.

RoPE integrates positional information directly into the attention mechanism by rotating the query and key vectors based on their positions. Specifically, the query ($q_i$) and key ($k_j$) vectors are derived by applying rotation matrices $R(i)$ and $R(j)$ to their respective vectors. The rotation matrices are designed to capture both the distance and direction between tokens:

$$
\text{Unnormalized Score}_{i,j} = (R(i)^T q_i) \cdot (R(j) k_j)
$$

Here, $q_i$ and $k_j$ are the query and key vectors from the linear projections $Q = XW_Q$ and $K = XW_K$. The terms $R(i)^T q_i$ and $R(j) k_j$ represent the rotated query and key vectors at positions $i$ and $j$, respectively. This unnormalized score is then passed through the softmax function to compute the final attention weights. By integrating relative positional information directly into the attention mechanism, this approach preserves the structure of the input sequence while offering flexibility for tasks that benefit from relative positioning.

### Respecting Causal Order

In generating text, it is crucial that the model respects the causal order of words. The Transformer achieves this through causal masking, which ensures that the model only considers past tokens when predicting the next one, maintaining the natural flow of language.

**Causal Masking**:  
Causal masking is implemented by masking out future tokens in the sequence, ensuring that each token is generated based solely on the preceding context. Mathematically, this is represented by a lower triangular matrix $M$, where the entry at position $(i, j)$ is 1 if $i \geq j$ and 0 otherwise:

$$
M = \begin{pmatrix}
1 & 0 & 0 & \cdots & 0 \\
1 & 1 & 0 & \cdots & 0 \\
1 & 1 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & 1 & 1 & \cdots & 1 \\
\end{pmatrix}
$$

When calculating the attention scores $\alpha_{i,j}$, this mask is applied to ensure that future tokens do not influence the prediction of the current token. The masked attention scores are defined as:

$$
\alpha_{i,j}^{\text{masked}} = M_{i,j} \cdot \alpha_{i,j}
$$

where $\alpha_{i,j}$ is the original attention score. By applying the mask $M$, the Transformer ensures that each token is influenced only by the tokens that precede it in the sequence. This modification is crucial for maintaining the temporal order of the sequence during text generation.

<a name="handling-the-curse-of-dimensionality"></a>
## Handling the Curse of Dimensionality

In this section, we will not introduce any new concepts or implementation details. Instead, our goal is to gain a deeper understanding of the attention mechanism itself. To achieve this, we will simplify the analysis by focusing specifically on the softmax component, denoted as $\alpha = \text{softmax}(XAX^T)$, where $A = W_Q W_K^T$. By doing so, we will set aside the $XW_V$ term and the scaling factor $\sqrt{d_x}$. We will also not discuss the effect of positional embedding and causal masking.

We will begin by visualizing how attention operates on a simple 2D Gaussian distribution, illustrating how the mechanism compresses data. This leads us to the concept of data lying on a low-dimensional manifold within a high-dimensional space. From there, we will connect these ideas to data covariance and the role of eigenvalues and eigenvectors, showing how $W_q W_Q^T$ functions similarly to PCA. Finally, we will explore the behavior of attention during training, particularly how it evolves from mean-pooling to more focused patterns as it learns to identify and utilize the underlying data structure.

### Visualization of a 2D Gaussian

To start, we visualize how the attention mechanism operates on a simple 2D Gaussian distribution. This basic scenario allows us to see the effects of attention in a more controlled and easily interpretable setting.

Imagine a dataset represented by a 2D Gaussian distribution centered at the origin. In this visualization, the attention mechanism, simplified to focus only on the softmax part, can be seen as compressing the data along certain directions. We can represent this transformation mathematically as:

$$
\mathbf{v}_{att} = c \cdot \mathbf{v}_x + (1 - c) \cdot \mathbf{v}_y
$$

where $\mathbf{v}_x$ and $\mathbf{v}_y$ are the base vectors along the $x$- and $y$-axes, respectively, and $c$ is a scalar coefficient ranging from 0 to 1.

**Figure 1** shows the original 2D Gaussian data and how it is transformed by the attention mechanism:

![image](https://hackmd.io/_uploads/ryTVpsE_C.png)
**Figure 1: Visualizing Dimensionality Reduction**. Blue: Original Data, Red: Transformed Data with $c = 0$ (focus on $y$-axis), Purple: Transformed Data with $c = 0.5$ (balance between $x$ and $y$), Green: Transformed Data with $c = 1$ (focus on $x$-axis).

The data starts as a full-rank, zero-centered Gaussian in a 2D space. The results show how the original data is "compressed" along directions determined by the linear combination of the base vectors. In this case, the base vectors are the *eigenvectors* of the data, and the spread of data along these directions are the associated *eigenvalues*. 

This compression, evident in this simplified 2D case, raises an important question: Is this compression meaningful or arbitrary? Since the original data distribution is full rank, with variance along both axes containing valuable information, any compression—regardless of the value of $c$—inevitably discards some of that information. 

However, in higher-dimensional spaces, such compression may be necessary to manage complexity and focus on the most relevant features. The compressed data in this example exhibits a pattern characteristic of a lower-dimensional manifold—essentially, a space where the data points lie along a simpler, more constrained shape within the higher-dimensional space. Many real-world datasets, even without any transformation, often reside on such manifolds. This concept will be explored further in the next section.

### Manifolds and Data Covariance

In the previous section, we observed how the attention mechanism can compress data into a pattern that lies on a lower-dimensional manifold. But what exactly is a manifold, and why is it significant in the context of high-dimensional data?

A manifold can be thought of as a lower-dimensional space embedded within a higher-dimensional one. For instance, consider a 2D surface, like the surface of a sphere, which exists within a 3D space. Imagine we curl up twist the plane the 2D Guassian data in earlier example, that will be a manifold.

In the context of machine learning, many real-world datasets naturally reside on manifolds. This means that while the data is represented in a high-dimensional space, its intrinsic structure is often much simpler, confined to a lower-dimensional surface within that space. Recognizing this underlying structure allows us to process and analyze the data more efficiently by focusing on the most informative aspects.

Note that we can always increase the feature space, adding more dimensions, to better capture the properties of the data. By expanding the feature space, we effectively allow the data to reside in a relatively lower-dimensional manifold within this expanded space. The challenge then becomes to identify and exploit this low-dimensional structure while managing the complexity introduced by the higher-dimensional space.

To quantitatively understand the structure of data on these manifolds, one of the tools we can use is the covariance matrix, denoted as $\Sigma_X$. The covariance matrix is computed as:

$$
\Sigma_X = \frac{1}{n} X^T X
$$

Here, $X$ represents the data matrix with dimensions $n \times d_x$, where $n$ is the number of data points and $d_x$ is the number of features. In the case of GPT-3, $d_x$ is 12,288. The resulting covariance matrix $\Sigma_X$ has dimensions $d_x \times d_x$, capturing the extent to which different features of the data vary together, revealing the directions along which the data has the most variance.

The covariance matrix can be decomposed into eigenvalues and eigenvectors through a process known as eigendecomposition:

$$
\Sigma_X = U \Lambda U^T
$$

where $U$ is a $d_x \times d_x$ matrix whose columns are the eigenvectors of $\Sigma_X$, and $\Lambda$ is a diagonal matrix containing the corresponding eigenvalues. 

- **Eigenvectors ($U$)**: These vectors represent the principal directions of variation in the data. Each eigenvector points along a direction where the data varies the most.
- **Eigenvalues ($\Lambda$)**: These values indicate the magnitude of variance in the direction of their corresponding eigenvectors. Larger eigenvalues correspond to directions with greater variance, meaning they capture more "information" about the data.

By focusing on the eigenvectors with the largest eigenvalues, we can effectively reduce the dimensionality of the data, concentrating on the directions that carry the most significant features. This process is closely related to Principal Component Analysis (PCA), a technique often used for dimensionality reduction.

In the context of the attention mechanism in Transformers, the matrix $W_Q W_K^T$—with the same size $d_x \times d_x$ as $\Sigma_X$—functions similarly to the covariance matrix. After training, this matrix should ideally capture variations in the data, analogous to how $\Sigma_X$ captures the principal components of the data. The attention mechanism can then assign higher attention scores to directions (eigenvectors) with larger eigenvalues, focusing on the most informative features and effectively managing the complexity of high-dimensional data.

### Pairwise Interactions and Eigenvectors

With the groundwork laid for understanding manifolds and data covariance, we can now delve into how the attention mechanism in Transformers operates at a more granular level. Specifically, we focus on the pairwise interactions between the query ($Q= XW_Q$) and key ($K= XW_K$) vectors, and how these interactions are influenced by the underlying data structure.

The attention mechanism computes attention scores using the matrix $QK^T$. This matrix can be understood as a sum of outer products of individual columns of $Q$ and $K$. Mathematically, this is expressed as:

$$
QK^T = \sum_{i=1}^{d_x} q_i k_i^T
$$

where $q_i$ and $k_i$ are the columns of $Q$ and $K$, respectively. The additivity of this operation allows us to analyze the interactions between individual pairs of columns, giving us insight into how the attention scores are formed.

Each column of $W_Q$ and $W_K$ can be expressed as a linear combination of the eigenvectors of the covariance matrix $\Sigma_X$. This is because the eigenvectors collectively form a valid coordiante system. That is, for any column $v_q$ from $W_Q$ and $v_k$ from $W_K$, we can write:

$$
v_q = \sum_i c_i e_i, \quad v_k = \sum_i d_i e_i
$$

where $e_i$ are the eigenvectors of $\Sigma_X$, and $c_i$, $d_i$ are the coefficients that determine how much of each eigenvector contributes to $v_q$ and $v_k$, respectively. If $X$ is fixed and known a priori, then it boils down to learn the coefficients (i.e. the array of $c$ and $d$ above).

### Analyzing the Impact of Eigenvectors

The eigenvector perspective allows us to analyze the impact of attention scores when $v_q$ and $v_k$ interact. These vectors are linear combinations of the eigenvectors of the covariance matrix. The total space of combinations is large, but we can analyze based on whether they are null eigenvectors (eigenvectors with zero eigenvalue, representing feature directions with no variance and thus no information) and their dominance degree (large or small eigenvalues).

The cases are as follows, illustrating that the predominant effects are mean-pooling and max-pooling, but these patterns exist in the transformed data rather than pooling onto a single token:
- **Null or Insignificant Eigenvectors**: If either $v_q$ or $v_k$ is a null or insignificant eigenvector, then they represent directions in $X$ where there is no or little variance. The outer product $X v_q (X v_k)^T$ will be close to zero, leading to a flat distribution when the softmax is applied. This means attention is evenly spread over all word tokens, effectively averaging their contributions. This scenario is akin to mean-pooling, where no particular token stands out in importance.
- **Both $v_q$ and $v_k$ are Non-Null Eigenvectors with Different Dominance Degrees**: When $v_q$ and $v_k$ are non-null eigenvectors but have different dominance degrees, the attention mechanism will highlight interactions between features captured by these eigenvectors. If $v_q$ is more dominant, tokens aligned with $v_q$ will have higher values, interacting strongly with other tokens in dimensions represented by $v_k$. This results in peaks in the attention map, indicating significant interactions between dominant and secondary features.
- **$v_q = v_k$ and are Non-Null Eigenvectors**: This is a special case where the peaks are diagonal. This results in a self-attention mechanism where tokens strongly attend to themselves. The distribution will be peaky, highlighting the most significant features captured by the principal eigenvector. Tokens representing key features or high variance will dominate the attention scores, making them stand out prominently.

![image](https://hackmd.io/_uploads/HJJm1EftR.png)

**Visualization**: The heatmaps above visualize the row-wise softmax of the outer product for different eigenvector combinations. We constructed the data to have a rank-2 covariance matrix by assigning specific eigenvalues to orthogonal eigenvectors. In the first case, $v_q$ is a dominant eigenvector and $v_k$ is a null eigenvector, resulting in a flat distribution. In the second case, both $v_q$ and $v_k$ are the same dominant eigenvector, producing diagonal peaks indicating self-attention. The third case shows $v_q$ as the first dominant eigenvector and $v_k$ as the second dominant, highlighting significant interactions between dominant and secondary features.

### Training Dynamics and Convergence

As large language models (LLMs) like Transformers undergo training, the attention mechanism evolves in its ability to focus on the most relevant features of the data. However, in the early stages of training, the attention mechanism often behaves in a way that resembles mean-pooling. This behavior can be understood by considering the relationship between the feature space and the effective dimensionality of the data.

Assume that the rank of the data's covariance matrix is 10% of the original feature size. For instance, in GPT-3, with a feature size of 12,288, the data manifold will effectively lie in a space of approximately 1,289 dimensions. Further assume that 10% of these non-null eigenvectors are insignificant, meaning they contribute little variance. Since 90% of the eigenvectors are null, the probability that the dominant components of $v_q$ and $v_k$ lie in the direction of null eigenvectors is approximately 98%. This includes the 81% probability for both components being null and the 18% probability for one component being null and the other non-null.

We studied training dynamics of [RoFormer](https://arxiv.org/abs/2104.09864) using 250M tokens from Wikipedia over near 10,000 steps. The model architecture consists of 6 layers, with a feature size of $d=384$ and the number of heads $h=6$. We examine the attention entropy variation of the heads during training and look at dimension-wise entropy distributions. To observe attention distributions, we used 1,000 random segments of length 512 for each checkpoint.

![image](https://hackmd.io/_uploads/Sk4M4gghA.png)

It is evident that the attention heads only seek out structure of data after certain steps, and that diversity emerge among them as the training progresses. 

![image](https://hackmd.io/_uploads/HJfnIxgnC.png)

We also look at dimension-wise entropy distributions, i.e. $\texttt{softmax}(q_i k_i^T/\sqrt d)$. We found they were uniform at the beginning and then approximate a scaling-law distribution (y-axis is log scale). Over 95.4% distributions have entropies higher than 6.21 even after 8k steps (The uniform distribution has an entropy of 6.24).

![image](https://hackmd.io/_uploads/ryQzDel2C.png)
Interestingly, for the head with low entropy, we find that dimension-wise attentions exhibiting high entropy (red distribution above). This is due to cumulative effects across dimensions, i.e. all the dimensions have slightly high concentration at the same location: for this particular data instance the attention head focuses on a token at a position in the middle of the sequence. Note softmax (green line) significantly promotes the unnormalized attention score (blue line)

It’s important to note that the input data $X$ itself is not static during training. As gradients propagate through the network, $X$ evolves along with the weight matrices. This dynamic adjustment means that the covariance structure of $X$ can change over time, further influencing the alignment of $W_Q$ and $W_K$. Initially, $X$ may exhibit a covariance structure that is less informative, but as training continues, the network learns to transform $X$ in ways that enhance its alignment with meaningful directions in the data space.

This interplay between the evolving $X$ and the refining $W_Q$ and $W_K$ leads to a progressive sharpening of the attention mechanism. The attention mechanism, which initially distributed attention evenly across tokens, now begins to prioritize tokens that align with significant features in the data. This evolution enhances the model’s ability to generalize and capture complex relationships within the data.

At convergence, the attention mechanism is highly specialized. It selectively attends to the most informative features of the input, ignoring directions in the data space that contribute little to the overall understanding. This specialization allows the model to efficiently manage the complexity of high-dimensional data, leading to better performance on a wide range of tasks.

In [22], the authors examine the converged behavior of BERT’s attention heads and find distinct patterns (replicated below). Specifically, in the shallow layers of the model, attention tends to be broad, which is indicated by higher entropy in the attention distributions. This means that in these layers, the data distribution has less structures. 
![image](https://hackmd.io/_uploads/B1_Tr7FY0.png)

Thus, while the intuitive explanation of the attention mechanism is that the learned feature of the query token seeks out information from contextual tokens via their keys, it’s crucial to understand that, fundamentally, these parameters—$X$'s embedding, $W_q$, and $W_k$—work in tandem to exploit the underlying structure of the data distribution.

<a name="capturing-diverse-relationships"></a>
## Capturing Diverse Relationships

In language modeling, capturing and understanding the diverse relationships within input text is crucial. These relationships can be syntactic, semantic, positional, or contextual, and a model's ability to grasp these nuances directly impacts its performance. Traditional models often struggled to balance these aspects, focusing too much on one type of relationship or failing to generalize across different contexts.

### Multi-Head Attention & Fractional Learning

Transformers address this challenge through **multi-head attention**, which allows the model to capture various types of relationships simultaneously. This mechanism can be viewed through the lens of **fractional learning**, where a complex task is divided into smaller, manageable parts that can be learned independently.

For instance, in the sentence "The cat quickly chased the mouse":
- One head might focus on the syntactic relationship between "cat" and "chased."
- Another might focus on the adverb "quickly."
- Yet another might consider the object "mouse."

Each head processes the input data in a separate subspace, capturing different aspects of the input. This parallel processing enables the model to build a more nuanced and comprehensive understanding of the input.

### Factorization Learning and Subspace Processing

Multi-head attention can also be understood as a form of **factorization learning**. Similar to how Convolutional Neural Networks (CNNs) use multiple filters to detect specific features, each attention head processes the input data within its own subspace, reducing complexity.

For an input vector $x \in \mathbb{R}^{12,288}$, the Transformer processes $x$ through multiple attention heads, each mapping to a smaller subspace, such as $\mathbb{R}^{128}$. The outputs are then concatenated to form a new representation of the input:

$$
x' = [x'_1, x'_2, \ldots, x'_{96}]
$$

where each $x'_i = \text{ATTN}_i(x)$. This approach reduces the dimensionality of the task, making it more tractable and efficient.

### Interaction Between Heads and Aggregation

The outputs from all attention heads are combined to form a unified representation. This is typically done by concatenating the head outputs, which are then processed by the Feed-Forward Network (FFN).

The effectiveness of multi-head attention relies on balancing **overlap** and **diversity** among the heads. If heads are too similar, they may redundantly focus on the same features, limiting the model's ability to generalize. Conversely, if heads are too disjoint, they might miss essential interactions between features. The combination of heads across layers allows for more complex interactions and hierarchical learning.

### Specialized Heads and Their Evolution During Training

As the model trains, attention heads may specialize in capturing specific types of relationships:
- Some heads might focus on syntactic structures.
- Others might specialize in semantic relationships.
- Positional information might be captured by particular heads.

![image](https://hackmd.io/_uploads/By-3BmtF0.png)

Empirical studies, such as the one conducted in [22], show that heads in a model like BERT focus on different patterns. This specialization allows the model to capture both broad and specific patterns, enhancing its ability to understand and generate language effectively.

<a name="handling-out-of-prompt-knowledge"></a>
## Handling Out-of-Prompt Knowledge

In many tasks, language models need to extend beyond the information explicitly provided in the input prompt. This ability to handle *out-of-prompt* knowledge is crucial for answering questions that require external information or making inferences based on world knowledge not present in the immediate context.

Tasks can be broadly classified into:
- **In-Prompt Tasks**: These tasks require the model to work solely with the information provided within the prompt. Examples include sentiment analysis, where the goal is to determine the sentiment expressed in a given text, and text classification, where the model assigns predefined labels to text based on its content.
- **Out-Prompt Tasks**: These tasks require the model to draw on knowledge that is not provided in the input prompt. For instance, answering the question "What is the capital of France?" requires the model to recall that the capital of France is Paris, a fact that may not be directly stated in the prompt.

Many tasks, such as translation or knowledge-based QA, involve both in-prompt and out-prompt components, making them hybrid tasks:
- **Translation**: Translation tasks involve transforming a sequence from one language to another, requiring extensive knowledge about the syntax, semantics, and idiomatic expressions of both languages. This internal knowledge helps generate accurate translations that are contextually and grammatically correct.
- **Knowledge-Based QA**: In these tasks, the model starts by mapping the prompt (query) to a retriever, and the retrieved contents become part of a new prompt. When multiple pieces of information need to be synthesized, or when internal reasoning is required, it aligns more with out-prompt tasks.
- **Medical Diagnosis**: While some information is provided in the prompt (e.g., symptoms and medical history), the model needs to draw upon extensive medical knowledge to make an accurate diagnosis, combining prompt analysis with knowledge retrieval.

### Limitations of the Attention Mechanism

The attention mechanism in Transformers is a powerful tool for capturing relationships within the input sequence. It works by computing a set of attention scores—essentially a heatmap—that determines how much each part of the input sequence should influence the output. These attention scores are then used to weigh the input vectors $X$, which are linearly transformed by the weight matrix $W_v$. The result is a weighted combination of the input, providing contextually relevant information for each position in the sequence.

However, the attention mechanism has significant limitations, particularly in its ability to store and utilize external knowledge:
- **Attention as a Heatmap**: The attention map is essentially a distribution imposed on $W_vX$. It tells the model how to distribute the influence of different parts of the input sequence, but it doesn't store any information itself.
- **Limited Storage Capacity of $W_v$**: The matrix $W_v$ is relatively small compared to the entire model, and its primary function is to transform the input into a format that can be weighted by the attention scores. It lacks the capacity to store significant amounts of external knowledge or contextual information.
- **No Memory Beyond Input Scope**: Since the attention mechanism depends entirely on the input provided during inference, it cannot incorporate or retrieve external knowledge dynamically. The attention mechanism effectively serves as a dynamic filter for the input but lacks the capability to act as a memory store.

### The Role of Feed-Forward Networks (FFN)

Given these limitations, Feed-Forward Networks (FFNs) in Transformers play a crucial role in handling out-prompt knowledge:

$$
\text{FFN}(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2
$$

In GPT-3, $W_1$ and $W_2$ are $(12288 \times 49152)$ and $(49152 \times 12288)$, respectively, forming an expand-compress configuration.

This structure, justified by Universal Approximation Theory, allows the FFN to function as a continuous, differentiable key-value store. The expanded hidden layer provides the flexibility needed to approximate complex function mappings, while the compression step retrieves relevant information, acting as a dynamic memory module.

![UAT](https://hackmd.io/_uploads/ryaHKSnsA.png)

**Figure**: The expand-compress structure of the FFN functions as a key-value store, where the expanded hidden layer allows for rich representation, and the compression layer retrieves relevant information based on the input.

This expand-compress configuration, coupled with ReLU activation, effectively implements a key-value store. The ReLU activation selectively activates certain "memories" stored in $W_2$, allowing the FFN to retrieve information relevant to the task at hand.

![image](https://hackmd.io/_uploads/Bkw_5suuR.png)

In this example, $W_2$ is a 4-by-8 matrix representing memory. The hidden layer output $h$ selectively activates elements, and $W_2h$ retrieves the corresponding information. This process mimics the retrieval of stored information based on the input key.

The expand-compress configuration of the FFN can also be viewed as a form of manifold learning, where the expansion maps the input to a higher-dimensional space to disentangle features, and the compression projects these features back onto a lower-dimensional manifold that captures the data's essential structure. This process allows the FFN to generalize knowledge, extract features, and perform implicit reasoning, making it crucial for handling out-prompt tasks.

#### Empirical Evidence

The expand-compress configuration of FFN was proposed in the original Transformer paper [5], where it was shown that if the hidden layer is made the same size as the input, performance degrades significantly. This suggests that the expansion step is crucial for accessing a richer representation space, allowing the model to capture more complex patterns and relationships in the data.

Since then, several studies have confirmed the importance of this configuration:
- **Xiong et al. [16]**: They demonstrated that removing the FFN leads to a substantial drop in performance. Their work suggested that the expansion phase allows the model to access a broader set of features, which is essential for capturing the nuances in the data.
- **Voita et al. [17]**: In their study, Voita and colleagues found that FFNs are less prunable than attention heads, indicating their crucial role in the model's overall performance. This finding highlights the importance of FFNs in retaining essential information that might be lost if the network were solely reliant on attention mechanisms.
- **Clark et al. [18]**: Their research emphasized the significance of FFNs in encoding linguistic information. They showed that FFNs contribute to the model's ability to understand and generate natural language by storing and retrieving linguistic patterns that are critical for language processing.
- **Roberts et al. [19]**: They focused on the role of FFNs in storing factual knowledge. Their findings indicated that the FFNs play a significant role in the model's ability to recall specific facts, which is particularly important for tasks requiring out-prompt knowledge retrieval.

A more in-depth analysis of FFNs in transformers was provided by **Geva et al. [20]** in their paper "Transformer Feed-Forward Layers Are Key-Value Memories." They showed that the first layer of the FFN acts as a key matching mechanism, where input activations are compared to learned keys. The ReLU activation then selects the most relevant keys. The second layer functions as a value retrieval mechanism, where the selected keys activate their associated values. This interpretation explains why increasing FFN size often improves model performance more than increasing the number of attention heads, as it directly increases the model's capacity to store and retrieve specific information.

### Learning FFNs with Next-Word Prediction

An intriguing finding from GPT-3 [7] is its ability to perform well on Closed Book Question Answering (QA) tasks—tasks that require out-prompt knowledge—despite being trained solely on next-word prediction. This raises an important question: How can a model trained this way demonstrate such a capability?

A likely explanation lies in GPT-3’s architecture, particularly its limited context window of 2048 tokens [7]. During pretraining, the model must predict the next word based on a restricted sequence of preceding tokens. Without the ability to rely on an extensive context, the model is forced to internalize broader knowledge within the FFNs. As the loss is backpropagated, essential world knowledge becomes embedded in the FFNs’ parameters. The finite context window thus acts as a catalyst, pushing the model to store and retrieve relevant information within its internal structures, enabling effective handling of out-prompt tasks, even in a zero-shot setting.

<a name="hierarchical-learning"></a>
## Iterative Refinement and Hierarchical Learning

Transformers utilize a deep architecture where each layer builds upon the previous one, allowing for **hierarchical learning**. This process of **iterative refinement** enables the model to progressively develop more abstract and comprehensive representations of the input data. The key idea is that each layer in the Transformer doesn’t just process the input data in isolation; rather, it enhances and refines the input by combining it with recalled knowledge and integrating increasingly complex patterns.

Before we delve deeper into the mechanisms of hierarchical learning, it’s important to highlight two critical aspects of how Transformers achieve this progressive refinement:
1. **BPE-Like Merging Behavior Through Attention**: The attention mechanism within each layer operates similarly to Byte Pair Encoding (BPE) in how it progressively merges and refines the input data. However, unlike BPE’s fixed tokenization process, the attention mechanism dynamically adjusts its understanding of the context with each subsequent layer. This continuous adaptation allows the model to capture increasingly complex relationships within the text, merging information in a manner that reflects the evolving context.
2. **Iterative Knowledge Integration Through FFN**: After the attention mechanism has transformed the input, the Feed-Forward Network (FFN) in each layer enriches this transformed data by integrating it with relevant contextual knowledge stored within that layer. This enriched data is then passed along to the next layer, where the process repeats. This iterative integration of knowledge ensures that the model continuously refines its understanding as the input progresses through the layers.


### Going Deeper with Skip Connections

Each Transformer block is composed of an attention layer followed by a feed-forward network (FFN) layer, each having a skip, or residual connection. The idea of skip connections was popularized by ResNet.

Given an arbitrary function $F(X)$, we can stack $n$ such layers to form a deeper network $T(X) = F_{n} \circ F_{n-1} \ldots \circ F_1(X)$.

From an optimization point of view, if $L$ is an implicit loss function, i.e., a hypothetical or abstract loss that captures the discrepancy or "error" in the representation at each layer, then:

$$
F_t(x_t) \approx -\eta \nabla L(x_t)
$$

and:

$$
x_{t+1} = x_t - \eta \nabla L(x_t) \approx x_t + F_t(x_t)
$$

Without skip connections, we would have a harder optimization problem to solve, i.e.,

$$
x_{t+1} = \arg \min_{x} L(x, x_t)
$$

Stacking transformer blocks with skip connections is critical because it enables the model to learn a residual function, which is an easier task. This approach improves gradient flow throughout the model, facilitating incremental refinement of features and enabling hierarchical feature learning. As a result, it allows for the construction of much deeper models, enhancing the model's capacity to capture complex patterns and dependencies in the data.

#### Stacking with Skip Connections

The Transformer model stacks multiple layers of attention and feed-forward networks, each with residual (or skip) connections. These connections allow the model to combine information from different depths, enabling the network to learn more robust representations while avoiding the vanishing gradient problem that often affects deep neural networks.

This hierarchical approach allows each layer to refine and build upon the representations learned by the previous layer, leading to a progressively more detailed understanding of the input data. Skip connections help maintain the flow of information throughout the network, ensuring that even very deep models can be trained effectively.

#### Attention and Feed-Forward Network Mechanisms

Earlier, we described the attention and feed-forward network (FFN) mechanisms in isolation:

- **Attention Mechanism**: Processes the input as $X'_{\text{attn}} = \big\|_{k=1}^{h} \text{ATTN}_k(X)$, where $\big\|$ denotes concatenation and $h$ is the number of attention heads.
- **Feed-Forward Network (FFN)**: Processes the input using two linear transformations with a ReLU activation in between: $\text{FFN}(X) = \text{max}(0, xW_1 + b_1)W_2 + b_2$, where $W_1$, $W_2$, $b_1$, and $b_2$ are learned parameters.

A Transformer layer connects an FFN to the output of the attention mechanism, incorporating layer normalization and skip connections to ensure stable training and better gradient flow. The complete formulation of one Transformer layer is as follows:

1. **Attention Layer**:
   - Normalize the input: $\text{LN}(X)$
   - Apply multi-head attention: $X'_{\text{attn}} = \big\|_{k=1}^{h} \text{ATTN}_k(\text{LN}(X))$
   - Add the residual connection: $X' = X + X'_{\text{attn}}$
2. **Feed-Forward Network (FFN) Layer**:
   - Normalize the output from the attention layer: $\text{LN}(X')$
   - Apply the feed-forward network: $X_{\text{ffn}} = \text{FFN}(\text{LN}(X'))$
   - Add the residual connection: $X_{i+1} = X' + X_{\text{ffn}}$

Combining these steps, one Transformer layer can be described as:
- $X'_{\text{attn}} = \big\|_{k=1}^{h} \text{ATTN}_k(\text{LN}(X))$
- $X' = X + X'_{\text{attn}}$
- $X_{\text{ffn}} = \text{FFN}(\text{LN}(X'))$
- $X_{i+1} = X' + X_{\text{ffn}}$

The following diagram shows these building blocks:
![skip-transformer](https://hackmd.io/_uploads/HyafKB3jR.png)

The entire Transformer model can be described as a composition of these layers:

$$
T(X) = F_{96} \circ F_{95} \circ \ldots \circ F_1(X)
$$

where each $F_i$ represents the $i$-th Transformer layer, incorporating the attention mechanism, feed-forward network, layer normalization, and skip connections as defined above.

### In-Model Normalization

Layer normalization is the key technique used in GPT-3. Unlike batch normalization, which normalizes across the batch dimension, layer normalization normalizes across the features within a single training example. This approach is particularly effective for stabilizing the training of transformers, which often process variable-length sequences.

Layer normalization is applied **before** the self-attention and feed-forward sub-layers within each Transformer block. This stabilizes the training of Transformers by normalizing the inputs to these sub-layers, reducing internal covariate shift and improving training stability. Mathematically, layer normalization can be expressed as:

$$
\text{LN}(x) = \frac{x - \mu}{\sigma + \epsilon} \gamma + \beta
$$

where $x$ is the input, $\mu$ and $\sigma$ are the mean and standard deviation of the input, $\epsilon$ is a small constant to prevent division by zero, and $\gamma$ and $\beta$ are learned parameters.

For each token, the mean and variance are computed as follows:

- **Mean**: The mean $\mu$ is computed over all the features of the input $x$:
  $$
  \mu = \frac{1}{d} \sum_{i=1}^{d} x_i
  $$
- **Variance**: The variance $\sigma^2$ is also computed over all the features of the input $x$:
  $$
  \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
  $$

Layer normalization normalizes the inputs to the self-attention and feed-forward sub-layers, helping to reduce internal covariate shift and improve training stability.

<a name="emerging-behavior-of-llms"></a>
## Emerging Behavior of LLMs

LLMs trained with the next-token prediction task exhibit remarkable emergent behaviors—abilities that arise naturally as a byproduct of scaling up model size and training on vast amounts of data. In this section, we introduce two of these emergent behaviors: In-Context Learning (ICL) and Chain-of-Thought (CoT) reasoning. Without any changes to the model’s parameters, and with only demonstration examles at inference time, ICL allows LLMs to perform new tasks, while CoT enables the model to solve complex, multi-step reasoning problems. These abilities are considered emergent because they were not explicitly programmed or trained into the models but rather appeared spontaneously as the models grew in size and complexity, as discussed in the "Emergent Abilities of Large Language Models" paper [30].

<a name="in-context-learning"></a>
### In-Context Learning

**Definition and Contrast with Traditional Few-Shot Learning:**  
Unlike traditional few-shot learning, where the model is fine-tuned on a small set of examples and its parameters are adjusted accordingly, ICL operates purely during inference. The model leverages patterns and relationships present in the prompt to generalize and solve the task at hand, making it a powerful tool for flexible and dynamic task adaptation.

**Example 1: Basic Arithmetic**

- **Input Examples:**
  - `2 + 3 = 5`
  - `4 + 7 = 11`
- **Query:**
  - `5 + 6 = ?`

**Example 2: Language Translation**

- **Input Examples:**
  - `English: "Hello" → Spanish: "Hola"`
  - `English: "Thank you" → Spanish: "Gracias"`
- **Query:**
  - `English: "Goodbye"`

In these examples, the model uses the demonstration examples in the prompt to infer a pattern and solve a new query based on that pattern. This ability to generalize from examples provided during inference—without updating its parameters—is what makes ICL distinct from traditional few-shot learning.

Understanding why LLMs exhibit ICL has been an active area of research. One compelling explanation is through the lens of Bayesian inference [26]. This perspective suggests that ICL can be viewed as the model performing implicit Bayesian inference, where the task itself is treated as a latent variable. This process is captured by the equation:

$$
P(y \mid x, C) = \sum_{\theta} P(y \mid x, \theta) P(\theta \mid C)
$$

- **$C$**: The context, or set of demonstration examples provided in the prompt.
- **$x$**: The new input (query) for which the model needs to predict an output.
- **$\theta$**: The latent task inferred from the context.
- **$P(\theta \mid C)$**: The model’s inferred probability distribution over possible tasks based on the context.
- **$P(y \mid x, \theta)$**: The probability of generating output $y$ given input $x$ under the inferred task $\theta$.

For instance, in the arithmetic example with the context `2 + 3 = 5` and `4 + 7 = 11`, and the query `5 + 6 = ?`, the model follows these steps:
- **Step 1: Task Inference ($\theta$):** The model uses the context $C$ to infer that the task ($\theta$) is addition. The probability $P(\theta \mid C)$ reflects this inference.
- **Step 2: Output Generation:** With the task inferred, the model calculates $P(y \mid x, \theta)$ for the query $x = 5 + 6$, leading to the output $y = 11$.

The GPT-3 paper [7] demonstrated that the capability of In-Context Learning (ICL) becomes significantly pronounced as model size increases. Larger models have more capacity to accurately infer tasks ($\theta$) from the context ($C$), thereby improving ICL performance across various tasks such as translation, arithmetic, and reasoning. This improvement is observed even when models with similar pretraining losses are compared, indicating that larger model sizes contribute to better generalization in ICL.

This Bayesian perspective highlights the model's ability to generalize tasks during inference. However, to fully understand how the model determines $\theta$ from the context, we can turn to another explanation involving **task vectors**, introduced in [27]. Task vectors are specific internal representations formed after processing the context. The hypothesis is that these task vectors effectively "select" the model's configuration or behavior that is appropriate for the inferred task $\theta$, guiding how the model processes the subsequent query.

In the study, task vectors are derived from the internal representations at a specific layer of the transformer model after processing the context $C$ using a "dummy query" ($x'$). This task vector represents the model's internal understanding of the task $\theta$. The use of a dummy query ensures that the task vector remains a pure reflection of the task derived solely from the context, without interference from the actual query.

Once formed, the task vector $v_{\theta}$ plays a crucial role in guiding the model's response to the actual query $x$. Similar tasks produce task vectors that cluster together in the model’s internal representation space, demonstrating that the model organizes and differentiates tasks effectively.

Inferring the right task $\theta$ depends on the structural similarity between the demonstration examples and the query. Transformer’s attention mechanism can easily identify and focus on this structural similarity, allowing it to lock onto a specific task $\theta$ during the inference process. However, for the model to solve a new query successfully, it's also crucial that both the query and the context originate from a similar distribution.

Studies such as [28] shows that the format of the demonstration examples plays a critical role in ICL’s success, emphasizing the importance of structural consistency. Moreover, the study reveals that as long as the input distribution between the context and query remains aligned, the model's performance remains robust—even when the labels in the demonstration examples are randomized. This underscores the importance of distributional alignment for effective task generalization.

<a name="chain-of-thought-cot-reasoning"></a>
### Chain-of-Thought (CoT) Reasoning

While In-Context Learning (ICL) can handle tasks like simple arithmetic or basic factual recall, it struggles with problems that require multi-step reasoning. For instance, consider the following problem:
- "If John has 3 apples and buys 2 more, then gives 1 to Jane, how many does he have left?"

In this scenario, ICL often falters because it, instead of a straightforward mapping from question to answer, solving the task requires understanding a sequence of actions and maintaining the state of the problem through each step. The model must correctly interpret each operation (addition, subtraction) and apply them in sequence, which is challenging for traditional ICL methods.

**Chain-of-Thought (CoT) prompting** [29] addresses this by introducing intermediate reasoning steps directly into the prompt. Instead of trying to infer the entire solution at once, CoT guides the model through the logical steps required to reach the correct answer. The demonstration to solve similar math problems becomes:
- **Input:**
  - "John has 3 apples."
  - "He buys 2 more, so now he has 5 apples."
  - "He gives 1 to Jane, leaving him with 4 apples."
- **Output:**
  - "John has 4 apples left."

Like ICL, CoT reasoning is an emergent behavior—it arises naturally without the need for explicit training or changes to model parameters. As the model's size increases, CoT's performance significantly improves, enabling the handling of more complex tasks. The key difference is that while ICL emerges as a general mechanism for task inference, CoT specifically addresses the challenge of multi-step reasoning.

The study demonstrates that CoT prompting enables LLMs to achieve state-of-the-art performance on a variety of reasoning tasks, such as math word problems and logical inference tasks, which were previously difficult to solve using standard ICL approaches.

As noted in the related work section of [29], the CoT approach is inspired by prior research showing the effectiveness of intermediate steps in solving complex problems. This raises the question of why a pretrained LLM at a large scale can be induced to exhibit such sophisticated reasoning behavior.

CoT can be understood as a method by which LLMs is induced to reveal the computation graph corresponding to solving a problem through a coherent sequence of reasoning steps. In [Part 1 of LLM and their Fundamental Limitations](https://hackmd.io/@LFNB9ifoT024aMHXU49sog/Bkh_RwLdC), we introduced the idea that LLMs learn a World Model by training on a corpus of texts that encode relational descriptions, causal inference, prediction, and common sense—all of which are graphical structures.

Thus, it is not surprising that LLMs can be nudged to reveal these underlying computation diagrams as explicit steps when solving a problem—particularly when given appropriate demonstrations. While this conjecture aligns with the observed behavior of LLMs under CoT prompting, proving it experimentally would be challenging.

### LLM as a Program

LLMs can be understood as executing different operations at varying levels of complexity and granularity. At the basic level, LLMs perform next-token prediction. However, this operation can be combined to handle more complex tasks, as seen in In-Context Learning (ICL) and Chain-of-Thought (CoT) reasoning, where basic operations are combined into more complex functions.

**Differentiating ICL and CoT**:  
ICL operates as a single-step function, where the model processes the prompt and returns a result based on recognized patterns. There is minimal need for maintaining intermediate states; the operation is direct and immediate.

In contrast, Chain-of-Thought (CoT) reasoning requires the model to maintain and update an internal state as it processes each step of a reasoning chain. This is similar to a program that follows a series of instructions, storing intermediate results before arriving at a final output. CoT involves the model following a sequence of logical steps, where each step builds on the previous one.

CoT reasoning involves maintaining and updating internal states throughout the process. This likely occurs through the model’s hidden states, which retain information across tokens and steps. The attention mechanism plays a crucial role here, allowing the model to focus on relevant parts of the input at each step.

A related approach was explored in [32], which introduces the concept of 'scratchpads' as placeholders for intermediate states during complex reasoning tasks. While CoT focuses on prompting the model to generate its own reasoning steps, the scratchpad approach provides an external memory mechanism for the model to show its work. 

**External Program Execution**:  
Instead of directly executing complex programs inside the LLM, it is often more effective to parse out the program and execute it externally, then integrate the results back into the LLM before generating a response. This approach allows for specialized processing and reduces the burden on the LLM to handle all aspects of computation internally. Two primary examples of this approach, both actively researched and adopted in practice, are Retrieval-Augmented Generation (RAG) and tool-calling:
- **Retrieval-Augmented Generation (RAG)**: RAG allows the model to retrieve relevant information from an external database before generating a response. This method is particularly useful for extending the model's knowledge with fresh or domain-specific information that wasn't part of the model's training data. By integrating these external resources, RAG enhances the LLM's ability to provide accurate, up-to-date, and contextually relevant responses [31].
- **Tool-calling**: Tool-calling involves the LLM using external tools or APIs to perform specific tasks, such as calculations, data processing, or accessing real-time information, before continuing with its main task. This approach is beneficial for tasks that require specialized knowledge or operations beyond the model’s internal capabilities, enabling more precise and practical outputs [33].

---

## References

1. Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A Neural Probabilistic Language Model. *Journal of Machine Learning Research*.
2. Elman, J. L. (1990). Finding structure in time. *Cognitive Science*.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
4. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*.
5. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
6. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.
7. Brown, T., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.
8. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *arXiv preprint arXiv:2203.02155*.
9. Christiano, P., et al. (2017). Deep reinforcement learning from human preferences. *NeurIPS*.
10. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*.
11. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1532-1543).
12. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1715-1725).
13. Fukushima, K. (1980). Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. Biological Cybernetics, 36(4), 193-202.
14. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
15. Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. Neural Networks, 2(5), 359-366.
16. Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., ... & Hu, X. (2020). On layer normalization in the transformer architecture. In International Conference on Machine Learning (pp. 10524-10533). PMLR.
17. Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 5797-5808).
18. Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). What does BERT look at? An analysis of BERT's attention. In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP (pp. 276-286).
19. Roberts, A., Raffel, C., & Shazeer, N. (2020). How much knowledge can you pack into the parameters of a language model? In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 5418-5426).
20. Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (pp. 5484-5495).
21. Eldan, R., & Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English?. arXiv preprint arXiv:2305.07759.
22. K. Clark, U. Khandelwal, O. Levy, and C. D. Manning, "What Does BERT Look At? An Analysis of BERT's Attention," in Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, 2019, pp. 276-286.
23. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. arXiv preprint arXiv:1607.06450.
24. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
25. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning (pp. 448-456). PMLR.
26. Xie, S. M., Li, F. L., & Ullrich, K. (2021). An Explanation of In-Context Learning as Implicit Bayesian Inference. arXiv preprint arXiv:2111.02080.
27. Hendel, R., Geva, M., & Globerson, A. (2023). In-Context Learning Creates Task Vectors. arXiv preprint arXiv:2310.15916.
28. Min, S., Wallace, E., Lee, H., & Hajishirzi, H. (2022). How Does In-Context Learning Work? A Framework for Understanding the Differences from Traditional Learning. arXiv preprint arXiv:2202.12837.
29. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *arXiv preprint arXiv:2201.11903*.
30. Wei, J., et al. (2022). "Emergent Abilities of Large Language Models." *arXiv preprint arXiv:2206.07682*.
31. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." arXiv preprint arXiv:2005.11401.
32. Nye, M., et al. (2021). "Show Your Work: Scratchpads for Intermediate Computation with Language Models. arXiv preprint arXiv:2112.00114.
33. Schick, T., et al. (2023). "Toolformer: Language Models Can Teach Themselves to Use Tools." arXiv preprint arXiv:2302.04761.


READ WITH CAUTION BEYOND THIS LINE; STILL EDITTING
