!pip install contractions
import pandas as pd
import random
import nltk
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import seaborn as sns
import re
import emoji

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Contractions dictionary
contractions = {
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
    "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
    "mustn't": "must not", "i'm": "i am", "you're": "you are", "he's": "he is", 
    "she's": "she is", "it's": "it is", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "we'd": "we would", "they'd": "they would", "i'll": "i will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will", "we'll": "we will", "they'll": "they will"
}

# Preprocessing Functions
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def remove_urls(text):
    return re.sub(r'http[s]?://\S+|www\.\S+', '', text)

def remove_mentions_hashtags(text):
    return re.sub(r'[@#]\w+', '', text)

def expand_contractions(text):
    for contraction, expanded in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', expanded, text, flags=re.IGNORECASE)
    return text

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def clean_text(text):
    text = remove_emojis(text)
    text = remove_urls(text)
    text = remove_mentions_hashtags(text)
    text = expand_contractions(text)
    text = remove_special_characters(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# EDA Helper Functions
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalpha()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words)-1)]
            synonyms = get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return ' '.join(new_words)
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, random_synonym)
    return ' '.join(new_words)

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        idx1 = random.randint(0, len(new_words)-1)
        idx2 = idx1
        counter = 0
        while idx2 == idx1:
            idx2 = random.randint(0, len(new_words)-1)
            counter += 1
            if counter > 3:
                return ' '.join(new_words)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return ' '.join(new_words)

def random_deletion(words, p):
    if len(words) == 1:
        return ' '.join(words)
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return words[rand_int]
    return ' '.join(new_words)

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    words = sentence.split(' ')
    words = [word for word in words if word != '']
    num_words = len(words)
    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1

    if alpha_sr > 0:
        n_sr = max(1, int(alpha_sr * num_words))
        for _ in range(num_new_per_technique):
            aug = synonym_replacement(words, n_sr)
            augmented_sentences.append(aug)

    if alpha_ri > 0:
        n_ri = max(1, int(alpha_ri * num_words))
        for _ in range(num_new_per_technique):
            aug = random_insertion(words, n_ri)
            augmented_sentences.append(aug)

    if alpha_rs > 0:
        n_rs = max(1, int(alpha_rs * num_words))
        for _ in range(num_new_per_technique):
            aug = random_swap(words, n_rs)
            augmented_sentences.append(aug)

    if p_rd > 0:
        for _ in range(num_new_per_technique):
            aug = random_deletion(words, p_rd)
            augmented_sentences.append(aug)

    random.shuffle(augmented_sentences)
    if len(augmented_sentences) > num_aug:
        augmented_sentences = augmented_sentences[:num_aug]
    elif len(augmented_sentences) < num_aug:
        augmented_sentences += [sentence] * (num_aug - len(augmented_sentences))
    return augmented_sentences

# Load the dataset
df = pd.read_csv('Tweets.csv')

# Select and rename columns
df = df[['text', 'airline_sentiment']]
df.rename(columns={'airline_sentiment': 'label'}, inplace=True)

# Convert labels to numerical values
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['label'].map(label_mapping)

# Apply preprocessing
df['text'] = df['text'].apply(clean_text)

# Remove empty texts after preprocessing
df = df[df['text'].str.strip() != ''].reset_index(drop=True)

# Get original class distribution
original_counts = df['label'].value_counts().sort_index()

# Augment minority classes
majority_count = original_counts.max()
minority_classes = original_counts[original_counts < majority_count].index
augmented_data = []

for label in minority_classes:
    df_min = df[df['label'] == label]
    num_to_generate = majority_count - len(df_min)
    samples = df_min['text'].tolist()
    generated_count = 0
    while generated_count < num_to_generate:
        text = random.choice(samples)
        aug_texts = eda(text, num_aug=1)
        aug_text_cleaned = clean_text(aug_texts[0])
        if aug_text_cleaned.strip() != '':
            augmented_data.append({'text': aug_text_cleaned, 'label': label})
            generated_count += 1

# Create augmented DataFrame and concatenate
df_aug = pd.DataFrame(augmented_data)
df_balanced = pd.concat([df, df_aug], ignore_index=True)

# Shuffle the balanced dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Get balanced class distribution
balanced_counts = df_balanced['label'].value_counts().sort_index()

# Save the balanced dataset
df_balanced.to_csv('balanced_tweets.csv', index=False)
print("Balanced dataset saved to 'balanced_tweets.csv'")

# Plotting the distributions
plt.figure(figsize=(12, 6))

# Original distribution
plt.subplot(1, 2, 1)
sns.barplot(x=original_counts.index, y=original_counts.values)
plt.title('Original Class Distribution')
plt.xlabel('Sentiment (0: Negative, 1: Neutral, 2: Positive)')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2], labels=['Negative', 'Neutral', 'Positive'], rotation=45)

# Balanced distribution
plt.subplot(1, 2, 2)
sns.barplot(x=balanced_counts.index, y=balanced_counts.values)
plt.title('Balanced Class Distribution')
plt.xlabel('Sentiment (0: Negative, 1: Neutral, 2: Positive)')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2], labels=['Negative', 'Neutral', 'Positive'], rotation=45)

plt.tight_layout()
plt.savefig('class_distribution_comparison.png')
plt.show()

print("Original class distribution:\n", original_counts)
print("Balanced class distribution:\n", balanced_counts)
