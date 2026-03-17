# 🧠 Mental Health Support Chatbot (Fine-Tuned)
### AI/ML Engineering Internship — Task 5 | DevelopersHub Corporation

---

## 📌 Task Objective
Build a chatbot that provides supportive and empathetic responses for stress, anxiety, and emotional wellness by fine-tuning a Large Language Model (LLM) on real human dialogue data.

---

## 📂 Dataset
- **Name:** EmpatheticDialogues (Facebook AI)
- **Source:** Kaggle — Empathetic Dialogues Facebook AI 25k
- **Total Rows:** 64,636
- **Cleaned Rows:** ~57,000 (after removing noise)
- **Unique Emotions:** 43 (anxious, sad, angry, guilty, hopeful, devastated, etc.)
- **Columns Used:** `emotion`, `empathetic_dialogues` (user), `labels` (counselor response)

### 🔍 Key Data Finding
Raw data contained `Customer :TEXT\nAgent :` format in dialogue column. Custom regex cleaning was applied to extract clean user messages and counselor responses before training.

---

## ⚙️ Preprocessing Steps
- Mounted Google Drive to load dataset in Colab
- Extracted clean text from `Customer :...\nAgent :` format using regex
- Removed `Agent :` / `Customer :` prefixes from labels (41 affected rows)
- Dropped rows with null emotion (4 rows)
- Filtered very short texts (< 4 words)
- Removed duplicate rows
- Set `padding_side = 'right'` for training, `'left'` for inference

---

## 🔧 Conversation Format
Each training sample was formatted as:
```
<|emotion|> {emotion} <|person|> {user message} <|bot|> {empathetic response} <|endoftext|>
```
This teaches the model to generate emotion-aware empathetic responses.

---

## 🤖 Model & Training

| Parameter | Value |
|-----------|-------|
| Base Model | GPT-Neo 125M (EleutherAI) |
| Training Samples | ~57,000 (full dataset) |
| Epochs | 3 |
| Batch Size | 8 (effective 16 with gradient accumulation) |
| Learning Rate | 3e-5 |
| LR Scheduler | Cosine |
| Max Sequence Length | 128 tokens |
| Mixed Precision | FP16 |
| Hardware | Google Colab T4 GPU |

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Unique Emotions Learned | 43 |
| Training Loss | Consistently decreasing |
| Eval Loss | Improving each epoch |
| Training Time | ~90-120 mins on T4 GPU |

---

## 🔍 Key Findings

1. **Data cleaning was critical** — raw `Customer:/Agent:` prefixes caused garbage output until properly removed with regex
2. **Padding direction matters** — `padding_side='right'` during training and `'left'` during inference was essential
3. **43 emotion categories** were learned — model generates different responses for `anxious` vs `excited` vs `devastated`
4. **Safety filter** immediately redirects crisis keywords to professional helplines
5. **Generation parameters** (temperature=0.6, repetition_penalty=1.4) keep responses focused and non-repetitive

---

## 💬 Sample Conversations

```
[Emotion: ANXIOUS]
User : I have been feeling really anxious lately and cannot sleep.
Bot  : That's terrible! I've been there before. It's so scary.

[Emotion: SAD]
User : I feel so lonely and like no one understands me.
Bot  : I'm so sorry to hear that. Have you tried talking to anyone?

[Emotion: HOPEFUL]
User : I started therapy and I think things might get better.
Bot  : That's wonderful! It takes a lot of courage to seek help.
```

---

## 🛡️ Safety Filter
```
User : I want to hurt myself
Bot  : I'm really concerned about what you've shared. Please reach out
       to a mental health professional immediately.
       In Pakistan: Umang helpline: 0317-4288665.
       You are not alone and help is available.
```

---

## 📈 Visualizations Generated
- Emotion category distribution (top 20 emotions)
- User message length distribution
- Bot response length distribution
- Training loss over steps
- Validation loss per epoch

---

## 🖥️ Chat Interface Features
- Emotion-aware responses (43 emotions supported)
- Switch emotion: `emotion:sad`
- View history: `history`
- Clear history: `clear`
- Exit: `quit`

---

## 🛠️ Tech Stack
- **Language:** Python 3.10
- **Libraries:** transformers, torch, accelerate, scikit-learn, datasets
- **Platform:** Google Colab (T4 GPU)
- **Base Model:** EleutherAI/gpt-neo-125m

---

## 🚀 How to Run
```bash
# 1. Clone the repository
git clone https://github.com/Dev-ZishanKhan/Mental_Health_Support_Chatbot

# 2. Open mental_health_chatbot.ipynb in Google Colab

# 3. Enable GPU: Runtime → Change runtime type → T4 GPU

# 4. Upload emotion-emotion_69k.csv to Google Drive

# 5. Replace HF token and CSV path in notebook

# 6. Run all cells top to bottom
```

---

## 📁 Repository Structure
```
mental-health-chatbot/
│
├── mental_health_chatbot.ipynb   # Main Jupyter notebook
├── README.md                     # Project documentation
└── images/                       # Generated plots
    ├── 01_emotion_distribution.png
    ├── 02_length_distribution.png
    └── 03_loss_curves.png
```

---

## 🔮 Possible Next Steps
- Train on full 64K dataset for richer responses
- Try Mistral-7B or LLaMA for production-grade quality
- Use LoRA fine-tuning for memory-efficient training
- Deploy via Streamlit or Gradio for web interface
- Push fine-tuned model to Hugging Face Hub
- Add multi-turn conversation memory

---

*DevelopersHub Corporation — AI/ML Engineering Internship 2026*