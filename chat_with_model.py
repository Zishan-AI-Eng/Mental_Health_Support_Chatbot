import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- Config ----
MODEL_PATH = 'mental_health_chatbot_full/'  # folder with your 6 files
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# ---- Load model ----
print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = 'left'
model = model.to(device)
model.eval()
print('Model loaded!')

def clean_response(text):
    for tag in ['<|emotion|>', '<|person|>', '<|bot|>', '<|endoftext|>']:
        text = text.replace(tag, '')
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 8]

    if sentences:
        # Pick the BEST sentence — longest one is usually most meaningful
        best = max(sentences[:3], key=len)
        if not best.endswith(('.', '!', '?')):
            best += '.'
        return best
    return text




def generate_response(user_message, emotion='anxious', max_new_tokens=80):
    # Safety filter
    crisis_keywords = [
        'suicide', 'kill myself', 'end my life', 'self harm',
        'hurt myself', 'want to die', 'overdose', 'end it all'
    ]
    if any(kw in user_message.lower() for kw in crisis_keywords):
        return (
            "I'm really concerned about what you've shared. "
            "Please reach out to a mental health professional immediately. "
            "Umang helpline (Pakistan): 0317-4288665. "
            "You are not alone."
        )

    prompt = f"<|emotion|> {emotion} <|person|> {user_message} <|bot|>"

    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=100
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens          = 60,    # shorter = more focused
            do_sample               = True,
            temperature             = 0.55,  # lower = less random
            top_p                   = 0.80,  # tighter nucleus
            top_k                   = 35,    # smaller vocab pool
            repetition_penalty      = 1.6,   # stronger = no repetition
            no_repeat_ngram_size    = 4,     # blocks 4-word repeated phrases
            pad_token_id            = tokenizer.eos_token_id,
            eos_token_id            = tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()

    cleaned = clean_response(generated)
    return cleaned if cleaned else "I hear you. What you're feeling is valid."

# ---- Chat loop ----
VALID_EMOTIONS = [
    'anxious', 'sad', 'angry', 'excited', 'grateful', 'surprised',
    'afraid', 'devastated', 'lonely', 'guilty', 'hopeful',
    'furious', 'disgusted', 'content', 'terrified', 'proud'
]

print('\n' + '=' * 60)
print('   🧠 MENTAL HEALTH SUPPORT CHATBOT')
print('   GPT-Neo 125M + EmpatheticDialogues')
print('=' * 60)
print('Commands: quit | history | clear | emotion:<name>')
print(f'Emotions : {", ".join(VALID_EMOTIONS[:8])}...')
print('=' * 60)

current_emotion = 'anxious'
history         = []

while True:
    try:
        user_input = input(f'\nYou [{current_emotion}]: ').strip()
    except (EOFError, KeyboardInterrupt):
        print('\nBot: Take care. Goodbye!')
        break

    if not user_input:
        continue

    if user_input.lower() in ['quit', 'exit', 'bye']:
        print('Bot: Thank you for talking with me. Take care!')
        break

    if user_input.lower() == 'history':
        if not history:
            print('No history yet.')
        else:
            for h in history:
                print(f'[{h["emotion"]}] You: {h["user"]}')
                print(f'Bot: {h["bot"]}\n')
        continue

    if user_input.lower() == 'clear':
        history = []
        print('History cleared.')
        continue

    if user_input.lower().startswith('emotion:'):
        current_emotion = user_input.split(':', 1)[1].strip().lower()
        current_emotion = current_emotion.replace('<','').replace('>','')
        print(f'Bot: Responding with {current_emotion} awareness now.')
        continue

    response = generate_response(user_input, current_emotion)
    print(f'\nBot: {response}')
    history.append({
        'emotion': current_emotion,
        'user'   : user_input,
        'bot'    : response
    })
