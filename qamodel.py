from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import threading

model = None
tokenizer = None
model_ready = False


def load_model():
    global model, tokenizer, model_ready
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model_ready = True
    print("QA model loaded successfully.")


# Start model loading in a background thread
threading.Thread(target=load_model, daemon=True).start()


def generate_answer(question: str, context: list) -> str:
    if not model_ready:
        return "Model is still loading. Please wait a few seconds and try again."

    full_context = " ".join(context)
    max_len = 512
    doc_stride = 50
    best_answer = ""
    best_score = float('-inf')

    # Tokenize question once
    question_tokens = tokenizer.tokenize(question)
    question_ids = tokenizer.convert_tokens_to_ids(question_tokens)
    question_len = len(question_ids)

    # Tokenize full context (without truncating)
    context_tokens = tokenizer.tokenize(full_context)

    # Sliding window
    for i in range(0, len(context_tokens), max_len - question_len - doc_stride):
        window_tokens = context_tokens[i:i + max_len - question_len - 3]  # [CLS] + Q + SEP + C + SEP

        tokens = tokenizer.build_inputs_with_special_tokens(
            tokenizer.convert_tokens_to_ids(question_tokens),
            tokenizer.convert_tokens_to_ids(window_tokens)
        )
        inputs = tokenizer.prepare_for_model(
            tokenizer.convert_tokens_to_ids(question_tokens),
            tokenizer.convert_tokens_to_ids(window_tokens),
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
            padding="max_length"
        )

        with torch.no_grad():
            outputs = model(**inputs)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Get best start-end pair for this window
        answer_start = torch.argmax(start_logits)
        answer_end = torch.argmax(end_logits)

        score = start_logits[0][answer_start] + end_logits[0][answer_end]

        if answer_end >= answer_start and score > best_score:
            best_score = score.item()
            answer_ids = inputs['input_ids'][0][answer_start:answer_end + 1]
            best_answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    question = question.lower()
    if not best_answer.strip() or best_answer == "<s>":
        greetings = {
            "hi": "Hi, how can I help you today?",
            "hello": "Hello! How can I help you today?",
            "bye": "Bye! Have a great day!",
            "goodbye": "Goodbye! Have a great day!",
            "thanks": "You're welcome! Let me know if you have more questions.",
            "thank you": "You're welcome!",
            "good morning": "Good morning! How can I help you today?",
            "good afternoon": "Good afternoon! How can I help you today?",
        }
        best_answer = greetings.get(question, "I don't know. I'll forward this to a human agent.")

    return best_answer

#test code
# folder_path = "testdata"

# documents = []
# for filename in os.listdir(folder_path):
#     if filename.endswith(".txt"):
#         with open(os.path.join(folder_path, filename), "r") as f:
#             documents.append(f.read())

# print(f"Loaded {len(documents)} documents.")


# user_question = input("Please enter your question: ")
# while user_question != "exit":
#     answer = generate_answer(user_question, documents)
#     print("Answer:", answer)
#     user_question = input("Please enter your question: ")

# print("Exiting...")